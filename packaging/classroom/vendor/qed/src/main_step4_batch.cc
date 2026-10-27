/*
 QEDispInv batch backend for AutoDispInv Step4.

 This executable intentionally computes only the Step4 forward/Jacobian blocks.
 The Python Step4 engine keeps ownership of the lateral joint inversion.
 */

#include "disp.hpp"
#include "swegn96.hpp"

#include <CLI11.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace Eigen;

namespace {

double brocher_dvp_dvs(double vs) {
  return 2.0947 - 2.0 * 0.8206 * vs + 3.0 * 0.2683 * vs * vs -
         4.0 * 0.0251 * vs * vs * vs;
}

double brocher_drho_dvp(double vp) {
  return 1.6612 - 2.0 * 0.4721 * vp + 3.0 * 0.0671 * vp * vp -
         4.0 * 0.0043 * vp * vp * vp +
         5.0 * 0.000106 * vp * vp * vp * vp;
}

ArrayXXd build_model(const ArrayXd &thick, const ArrayXd &vp,
                     const ArrayXd &vs, const ArrayXd &rho) {
  const int nl = thick.rows();
  ArrayXXd model(nl, 5);
  double z = 0.0;
  for (int i = 0; i < nl; ++i) {
    model(i, 0) = static_cast<double>(i + 1);
    model(i, 1) = z;
    model(i, 2) = rho(i);
    model(i, 3) = vs(i);
    model(i, 4) = vp(i);
    if (i < nl - 1) {
      z += thick(i);
    }
  }
  return model;
}

void write_value(std::ostream &out, double value) {
  if (std::isfinite(value)) {
    out << std::setprecision(17) << value;
  } else {
    out << "nan";
  }
}

} // namespace

int main(int argc, char const *argv[]) {
  CLI::App app{"Batch Step4 forward and analytic Jacobian backend."};
  std::string file_input;
  std::string file_output;
  app.add_option("-i,--input", file_input, "batch input text file")->required();
  app.add_option("-o,--output", file_output, "batch output text file")
      ->required();
  CLI11_PARSE(app, argc, argv);

  std::ifstream in(file_input);
  if (!in) {
    throw std::runtime_error("Cannot open input file: " + file_input);
  }

  int nstation = 0;
  int nlayer = 0;
  int nperiod = 0;
  int mode = 0;
  int sh_flag = 0;
  int compute_kernel_flag = 1;
  in >> nstation >> nlayer >> nperiod >> mode >> sh_flag >> compute_kernel_flag;
  if (!in || nstation <= 0 || nlayer <= 0 || nperiod <= 0 || mode < 0) {
    throw std::runtime_error("Invalid Step4 batch header.");
  }

  ArrayXd periods(nperiod);
  for (int i = 0; i < nperiod; ++i) {
    in >> periods(i);
  }

  std::vector<ArrayXd> thick_list(nstation, ArrayXd(nlayer));
  std::vector<ArrayXd> vp_list(nstation, ArrayXd(nlayer));
  std::vector<ArrayXd> vs_list(nstation, ArrayXd(nlayer));
  std::vector<ArrayXd> rho_list(nstation, ArrayXd(nlayer));
  for (int ix = 0; ix < nstation; ++ix) {
    int station_id = -1;
    in >> station_id;
    if (!in || station_id != ix) {
      throw std::runtime_error("Invalid or out-of-order station block.");
    }
    for (int il = 0; il < nlayer; ++il)
      in >> thick_list[ix](il);
    for (int il = 0; il < nlayer; ++il)
      in >> vp_list[ix](il);
    for (int il = 0; il < nlayer; ++il)
      in >> vs_list[ix](il);
    for (int il = 0; il < nlayer; ++il)
      in >> rho_list[ix](il);
  }
  if (!in) {
    throw std::runtime_error("Failed while reading Step4 batch input.");
  }

  std::vector<ArrayXd> cg_list(nstation, ArrayXd::Constant(nperiod, NAN));
  std::vector<ArrayXXd> kernel_list(
      nstation, ArrayXXd::Zero(nperiod, nlayer));

  const bool sh = sh_flag != 0;
  const bool compute_kernel = compute_kernel_flag != 0;

#if defined(_OPENMP)
#pragma omp parallel for if (!compute_kernel)
#endif
  for (int ix = 0; ix < nstation; ++ix) {
    ArrayXXd model =
        build_model(thick_list[ix], vp_list[ix], vs_list[ix], rho_list[ix]);
    Dispersion disp(model, sh);
    SwEgn96 se(model, sh);

    for (int ip = 0; ip < nperiod; ++ip) {
      const double period = periods(ip);
      if (!(period > 0.0) || !std::isfinite(period)) {
        continue;
      }
      const double freq = 1.0 / period;
      const double c = disp.search_mode(freq, mode);
      cg_list[ix](ip) = c;
      if (!compute_kernel || !std::isfinite(c) || c == 0.0) {
        continue;
      }

      auto ker = se.kernel(freq, c);
      const ArrayXd &dc_dvp = ker["vp"];
      const ArrayXd &dc_dvs = ker["vs"];
      const ArrayXd &dc_drho = ker["rho"];
      for (int il = 0; il < nlayer; ++il) {
        const double vs = vs_list[ix](il);
        const double vp = vp_list[ix](il);
        const double dvp_dvs = brocher_dvp_dvs(vs);
        const double drho_dvp = brocher_drho_dvp(vp);
        const double dc_dvs_total =
            dc_dvs(il) + dc_dvp(il) * dvp_dvs +
            dc_drho(il) * drho_dvp * dvp_dvs;
        kernel_list[ix](ip, il) = (vs / c) * dc_dvs_total;
      }
    }
  }

  std::ofstream out(file_output);
  if (!out) {
    throw std::runtime_error("Cannot open output file: " + file_output);
  }
  out << nstation << " " << nperiod << " " << nlayer << "\n";
  for (int ix = 0; ix < nstation; ++ix) {
    out << "station " << ix << "\n";
    out << "cg";
    for (int ip = 0; ip < nperiod; ++ip) {
      out << " ";
      write_value(out, cg_list[ix](ip));
    }
    out << "\n";
    out << "kernel\n";
    for (int ip = 0; ip < nperiod; ++ip) {
      for (int il = 0; il < nlayer; ++il) {
        if (il > 0) {
          out << " ";
        }
        write_value(out, kernel_list[ix](ip, il));
      }
      out << "\n";
    }
  }
  return 0;
}
