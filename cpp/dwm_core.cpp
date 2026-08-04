#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace py = pybind11;
using Complex = std::complex<double>;

struct Matrix {
    std::size_t rows;
    std::size_t cols;
    std::vector<Complex> values;

    Matrix(std::size_t row_count, std::size_t column_count)
        : rows(row_count), cols(column_count), values(row_count * column_count, Complex(0.0, 0.0)) {}

    Complex& operator()(std::size_t row, std::size_t column) {
        return values[row * cols + column];
    }
    const Complex& operator()(std::size_t row, std::size_t column) const {
        return values[row * cols + column];
    }
};

struct VerticalFkResponse {
    std::vector<double> frequency;
    std::vector<double> wavenumber;
    std::vector<Complex> hr;
    std::vector<Complex> ht;
    std::vector<Complex> hv;
    std::vector<Complex> vr;
    std::vector<Complex> vv;
};

static std::pair<Complex, Complex> stable_sh_ch(const Complex& value) {
    const double real_part = value.real();
    const double imaginary_part = value.imag();
    const Complex imaginary_unit(0.0, 1.0);
    Complex sh;
    Complex ch;
    if (real_part >= 0.0) {
        const double decay = std::exp(-2.0 * real_part);
        sh = ((1.0 - decay) * std::cos(imaginary_part)
              + (1.0 + decay) * std::sin(imaginary_part) * imaginary_unit) / 2.0;
        ch = ((1.0 + decay) * std::cos(imaginary_part)
              + (1.0 - decay) * std::sin(imaginary_part) * imaginary_unit) / 2.0;
    } else {
        const double growth = std::exp(2.0 * real_part);
        sh = -((1.0 - growth) * std::cos(imaginary_part)
               - (1.0 + growth) * std::sin(imaginary_part) * imaginary_unit) / 2.0;
        ch = ((1.0 + growth) * std::cos(imaginary_part)
              - (1.0 - growth) * std::sin(imaginary_part) * imaginary_unit) / 2.0;
    }
    return {sh, ch};
}

static void conjugate_in_place(Matrix& matrix) {
    for (Complex& value : matrix.values) {
        value = std::conj(value);
    }
}

static std::pair<Matrix, Matrix> stiff_element(
    double cp_input,
    double cs_input,
    double damp_p,
    double damp_s,
    double density,
    double thickness,
    double signed_wavenumber,
    Complex omega_input
) {
    const double wavenumber = std::abs(signed_wavenumber);
    const double omega_real = omega_input.real();
    double omega_imaginary = omega_input.imag();
    if (omega_imaginary > 0.0) {
        omega_imaginary = -omega_imaginary;
    }
    Complex omega(std::abs(omega_real), omega_imaginary);
    const double frequency_sign = (omega_real > 0.0) - (omega_real < 0.0);
    const Complex cp = cp_input * Complex(1.0, damp_p * frequency_sign);
    const Complex cs = cs_input * Complex(1.0, damp_s * frequency_sign);
    const Complex a2 = (cs / cp) * (cs / cp);
    const Complex shear_modulus = cs * cs * density;
    Complex kp = std::sqrt(Complex(wavenumber * wavenumber, 0.0) - (omega / cp) * (omega / cp));
    if (kp.real() < 0.0) kp = -kp;
    Complex ks = std::sqrt(Complex(wavenumber * wavenumber, 0.0) - (omega / cs) * (omega / cs));
    if (ks.real() < 0.0) ks = -ks;
    Complex p(0.0, 0.0);
    Complex s(0.0, 0.0);
    if (wavenumber != 0.0) {
        p = std::sqrt(Complex(1.0, 0.0) - std::pow(omega / (wavenumber * cp), 2.0));
        s = std::sqrt(Complex(1.0, 0.0) - std::pow(omega / (wavenumber * cs), 2.0));
        if (p.real() < 0.0) p = -p;
        if (s.real() < 0.0) s = -s;
    }

    const bool halfspace = std::isinf(thickness);
    Matrix k_svp(halfspace ? 2 : 4, halfspace ? 2 : 4);
    Matrix k_sh(halfspace ? 1 : 2, halfspace ? 1 : 2);
    const Complex zero(0.0, 0.0);
    if (halfspace) {
        if (wavenumber > 0.0 && omega != zero) {
            k_sh(0, 0) = ks * shear_modulus;
            const Complex scalar = 2.0 * wavenumber * shear_modulus;
            const Complex factor = (1.0 - s * s) / (2.0 * (1.0 - p * s));
            k_svp(0, 0) = scalar * factor * p;
            k_svp(0, 1) = scalar * (-factor + 1.0);
            k_svp(1, 0) = scalar * (-factor + 1.0);
            k_svp(1, 1) = scalar * factor * s;
        } else if (wavenumber > 0.0 && omega == zero) {
            k_sh(0, 0) = wavenumber * shear_modulus;
            const Complex scalar = 2.0 * wavenumber * shear_modulus / (1.0 + a2);
            k_svp(0, 0) = scalar;
            k_svp(0, 1) = scalar * a2;
            k_svp(1, 0) = scalar * a2;
            k_svp(1, 1) = scalar;
        } else if (wavenumber == 0.0 && omega != zero) {
            k_sh(0, 0) = Complex(0.0, 1.0) * omega * density * cs;
            k_svp(0, 0) = Complex(0.0, 1.0) * omega * density * cs;
            k_svp(1, 1) = Complex(0.0, 1.0) * omega * density * cp;
        }
    } else if (wavenumber > 0.0 && omega != zero) {
        auto [sh_s, ch_s] = stable_sh_ch(ks * thickness);
        auto [sh_p, ch_p] = stable_sh_ch(kp * thickness);
        const Complex coth_s = ch_s / sh_s;
        const double abs_real_s = std::abs((ks * thickness).real());
        const double abs_real_p = std::abs((kp * thickness).real());
        const double kh = wavenumber * thickness;
        const Complex ps = p * s;
        Complex denominator;
        if (p == zero) {
            denominator = 2.0 * (std::exp(-abs_real_s) - ch_s) + sh_s * kh / s;
        } else if (s == zero) {
            denominator = 2.0 * (std::exp(-abs_real_p) - ch_p) + sh_p * kh / p;
        } else {
            denominator = 2.0 * (std::exp(-(abs_real_s + abs_real_p)) - ch_p * ch_s)
                        + (1.0 / ps + ps) * sh_p * sh_s;
        }
        const Complex chp_chs_d = ch_p * ch_s / denominator;
        const Complex chp_shs_d = ch_p * sh_s / denominator;
        const Complex shp_chs_d = sh_p * ch_s / denominator;
        const Complex shp_shs_d = sh_p * sh_s / denominator;
        const Complex chp_d = std::exp(-abs_real_s) * ch_p / denominator;
        const Complex shp_d = std::exp(-abs_real_s) * sh_p / denominator;
        const Complex chs_d = std::exp(-abs_real_p) * ch_s / denominator;
        const Complex shs_d = std::exp(-abs_real_p) * sh_s / denominator;
        const Complex um_d = std::exp(-abs_real_p - abs_real_s) / denominator;
        sh_s *= std::exp(abs_real_s);
        k_sh(0, 0) = ks * shear_modulus * coth_s;
        k_sh(0, 1) = -ks * shear_modulus / sh_s;
        k_sh(1, 0) = k_sh(0, 1);
        k_sh(1, 1) = k_sh(0, 0);
        const Complex s1 = wavenumber * shear_modulus * (1.0 - s * s);
        const Complex s2 = wavenumber * shear_modulus * (1.0 + s * s);
        Complex k11;
        Complex k12 = s1 * (um_d - chp_chs_d) + s2;
        Complex k22;
        Matrix k_cross(2, 2);
        if (p == zero) {
            k11 = s1 * chp_shs_d / s;
            k22 = s1 * (kh * chs_d - s * chp_shs_d);
            k_cross(0, 0) = s1 * (-shs_d / s);
            k_cross(0, 1) = s1 * (chp_d - chs_d);
            k_cross(1, 0) = -s1 * (chp_d - chs_d);
            k_cross(1, 1) = s1 * (s * shs_d - kh * um_d);
        } else if (s == zero) {
            k11 = s1 * (kh * chp_d - p * shp_chs_d);
            k22 = s1 * shp_chs_d / p;
            k_cross(0, 0) = s1 * (p * shp_d - kh * um_d);
            k_cross(0, 1) = s1 * (chp_d - chs_d);
            k_cross(1, 0) = -s1 * (chp_d - chs_d);
            k_cross(1, 1) = -s1 * shp_d / p;
        } else {
            k11 = s1 * (chp_shs_d - ps * shp_chs_d) / s;
            k12 = s1 * (um_d - chp_chs_d + ps * shp_shs_d) + s2;
            k22 = s1 * (shp_chs_d - ps * chp_shs_d) / p;
            k_cross(0, 0) = s1 * (ps * shp_d - shs_d) / s;
            k_cross(0, 1) = s1 * (chp_d - chs_d);
            k_cross(1, 0) = -s1 * (chp_d - chs_d);
            k_cross(1, 1) = s1 * (ps * shs_d - shp_d) / p;
        }
        k_svp(0, 0) = k11;
        k_svp(0, 1) = k12;
        k_svp(1, 0) = k12;
        k_svp(1, 1) = k22;
        k_svp(2, 2) = k11;
        k_svp(2, 3) = -k12;
        k_svp(3, 2) = -k12;
        k_svp(3, 3) = k22;
        for (std::size_t row = 0; row < 2; ++row) {
            for (std::size_t column = 0; column < 2; ++column) {
                k_svp(row, column + 2) = k_cross(row, column);
                k_svp(row + 2, column) = k_cross(column, row);
            }
        }
    } else if (wavenumber > 0.0 && omega == zero) {
        const double kh = wavenumber * thickness;
        auto [sh, ch] = stable_sh_ch(Complex(kh, 0.0));
        const Complex coth = ch / sh;
        const double abs_real = std::abs(kh);
        const Complex denominator = std::pow(1.0 + a2, 2.0) * sh * sh
                                  - kh * kh * std::pow(1.0 - a2, 2.0)
                                    * std::exp(-2.0 * abs_real);
        const Complex sh_sh_d = sh * sh / denominator;
        const Complex ch_sh_d = ch * sh / denominator;
        const Complex ch_d = ch * std::exp(-abs_real) / denominator;
        const Complex sh_d = sh * std::exp(-abs_real) / denominator;
        const Complex um_d = std::exp(-2.0 * abs_real) / denominator;
        sh *= std::exp(abs_real);
        k_sh(0, 0) = wavenumber * shear_modulus * coth;
        k_sh(0, 1) = -wavenumber * shear_modulus / sh;
        k_sh(1, 0) = k_sh(0, 1);
        k_sh(1, 1) = k_sh(0, 0);
        Matrix k11(2, 2);
        const Complex scalar = 2.0 * wavenumber * shear_modulus;
        k11(0, 0) = scalar * ((1.0 + a2) * ch_sh_d - kh * (1.0 - a2) * um_d);
        k11(0, 1) = scalar * (-(1.0 + a2) * sh_sh_d + 1.0);
        k11(1, 0) = k11(0, 1);
        k11(1, 1) = scalar * ((1.0 + a2) * ch_sh_d + kh * (1.0 - a2) * um_d);
        Matrix k_cross(2, 2);
        k_cross(0, 0) = scalar * (kh * (1.0 - a2) * ch_d - (1.0 + a2) * sh_d);
        k_cross(0, 1) = scalar * kh * (1.0 - a2) * sh_d;
        k_cross(1, 0) = -k_cross(0, 1);
        k_cross(1, 1) = -scalar * (kh * (1.0 - a2) * ch_d + (1.0 + a2) * sh_d);
        for (std::size_t row = 0; row < 2; ++row) {
            for (std::size_t column = 0; column < 2; ++column) {
                k_svp(row, column) = k11(row, column);
                k_svp(row + 2, column + 2) = k11(row, column) * ((row != column) ? -1.0 : 1.0);
                k_svp(row, column + 2) = k_cross(row, column);
                k_svp(row + 2, column) = k_cross(column, row);
            }
        }
    } else if (wavenumber == 0.0 && omega != zero) {
        const Complex omega_p = omega * thickness / cp;
        const Complex omega_s = omega * thickness / cs;
        const Complex sh_scale = density * cs * omega;
        k_sh(0, 0) = sh_scale * std::cos(omega_s) / std::sin(omega_s);
        k_sh(0, 1) = -sh_scale / std::sin(omega_s);
        k_sh(1, 0) = k_sh(0, 1);
        k_sh(1, 1) = k_sh(0, 0);
        const Complex scale = density * omega;
        k_svp(0, 0) = scale * cs * std::cos(omega_s) / std::sin(omega_s);
        k_svp(0, 2) = -scale * cs / std::sin(omega_s);
        k_svp(1, 1) = scale * cp * std::cos(omega_p) / std::sin(omega_p);
        k_svp(1, 3) = -scale * cp / std::sin(omega_p);
        k_svp(2, 0) = k_svp(0, 2);
        k_svp(2, 2) = k_svp(0, 0);
        k_svp(3, 1) = k_svp(1, 3);
        k_svp(3, 3) = k_svp(1, 1);
    } else {
        const Complex inverse_a2 = 1.0 / a2;
        const Complex sh_scale = shear_modulus / thickness;
        k_sh(0, 0) = sh_scale;
        k_sh(0, 1) = -sh_scale;
        k_sh(1, 0) = -sh_scale;
        k_sh(1, 1) = sh_scale;
        k_svp(0, 0) = sh_scale;
        k_svp(0, 2) = -sh_scale;
        k_svp(1, 1) = sh_scale * inverse_a2;
        k_svp(1, 3) = -sh_scale * inverse_a2;
        k_svp(2, 0) = -sh_scale;
        k_svp(2, 2) = sh_scale;
        k_svp(3, 1) = -sh_scale * inverse_a2;
        k_svp(3, 3) = sh_scale * inverse_a2;
    }

    if (signed_wavenumber < 0.0) {
        if (k_svp.rows > 2) {
            k_sh(0, 1) = -k_sh(0, 1);
            k_sh(1, 0) = -k_sh(1, 0);
        }
        for (std::size_t index = 1; index < k_svp.rows; index += 2) {
            for (std::size_t column = 0; column < k_svp.cols; ++column) {
                k_svp(index, column) = -k_svp(index, column);
            }
            for (std::size_t row = 0; row < k_svp.rows; ++row) {
                k_svp(row, index) = -k_svp(row, index);
            }
        }
    }
    if (omega_real < 0.0) {
        conjugate_in_place(k_sh);
        conjugate_in_place(k_svp);
    }
    return {k_svp, k_sh};
}

static std::pair<Matrix, Matrix> global_stiffness(
    const std::vector<double>& vp,
    const std::vector<double>& vs,
    const std::vector<double>& damp_p,
    const std::vector<double>& damp_s,
    const std::vector<double>& density,
    const std::vector<double>& thickness,
    double wavenumber,
    Complex omega
) {
    const std::size_t layer_count = vp.size();
    if (layer_count == 0 || vs.size() != layer_count || damp_p.size() != layer_count
        || damp_s.size() != layer_count || density.size() != layer_count
        || thickness.size() != layer_count) {
        throw std::invalid_argument("all model arrays must have equal non-zero length");
    }
    Matrix global_sh(layer_count, layer_count);
    Matrix global_svp(2 * layer_count, 2 * layer_count);
    for (std::size_t layer = 0; layer + 1 < layer_count; ++layer) {
        auto [element_svp, element_sh] = stiff_element(
            vp[layer], vs[layer], damp_p[layer], damp_s[layer], density[layer],
            thickness[layer], wavenumber, omega
        );
        for (std::size_t row = 0; row < 2; ++row) {
            for (std::size_t column = 0; column < 2; ++column) {
                global_sh(layer + row, layer + column) += element_sh(row, column);
            }
        }
        for (std::size_t row = 0; row < 4; ++row) {
            for (std::size_t column = 0; column < 4; ++column) {
                global_svp(2 * layer + row, 2 * layer + column) += element_svp(row, column);
            }
        }
    }
    const std::size_t layer = layer_count - 1;
    auto [element_svp, element_sh] = stiff_element(
        vp[layer], vs[layer], damp_p[layer], damp_s[layer], density[layer],
        thickness[layer], wavenumber, omega
    );
    global_sh(layer, layer) += element_sh(0, 0);
    for (std::size_t row = 0; row < 2; ++row) {
        for (std::size_t column = 0; column < 2; ++column) {
            global_svp(2 * layer + row, 2 * layer + column) += element_svp(row, column);
        }
    }
    return {global_sh, global_svp};
}

static std::vector<Complex> solve_linear(Matrix matrix, std::vector<Complex> right_hand_side) {
    if (matrix.rows != matrix.cols || right_hand_side.size() != matrix.rows) {
        throw std::invalid_argument("linear solve dimensions do not match");
    }
    const std::size_t n = matrix.rows;
    for (std::size_t pivot_column = 0; pivot_column < n; ++pivot_column) {
        std::size_t pivot_row = pivot_column;
        double pivot_size = std::abs(matrix(pivot_row, pivot_column));
        for (std::size_t row = pivot_column + 1; row < n; ++row) {
            const double candidate_size = std::abs(matrix(row, pivot_column));
            if (candidate_size > pivot_size) {
                pivot_size = candidate_size;
                pivot_row = row;
            }
        }
        if (pivot_size == 0.0 || !std::isfinite(pivot_size)) {
            throw std::runtime_error("singular or non-finite stiffness matrix");
        }
        if (pivot_row != pivot_column) {
            for (std::size_t column = pivot_column; column < n; ++column) {
                std::swap(matrix(pivot_column, column), matrix(pivot_row, column));
            }
            std::swap(right_hand_side[pivot_column], right_hand_side[pivot_row]);
        }
        for (std::size_t row = pivot_column + 1; row < n; ++row) {
            const Complex factor = matrix(row, pivot_column) / matrix(pivot_column, pivot_column);
            matrix(row, pivot_column) = Complex(0.0, 0.0);
            for (std::size_t column = pivot_column + 1; column < n; ++column) {
                matrix(row, column) -= factor * matrix(pivot_column, column);
            }
            right_hand_side[row] -= factor * right_hand_side[pivot_column];
        }
    }
    std::vector<Complex> solution(n);
    for (std::size_t reverse = 0; reverse < n; ++reverse) {
        const std::size_t row = n - 1 - reverse;
        Complex value = right_hand_side[row];
        for (std::size_t column = row + 1; column < n; ++column) {
            value -= matrix(row, column) * solution[column];
        }
        solution[row] = value / matrix(row, row);
    }
    return solution;
}

static VerticalFkResponse fk_response(
    const std::vector<double>& vp,
    const std::vector<double>& vs,
    const std::vector<double>& damp_p,
    const std::vector<double>& damp_s,
    const std::vector<double>& density,
    const std::vector<double>& thickness,
    std::size_t sample_count,
    double dt,
    double maximum_distance,
    double delay,
    double maximum_frequency,
    std::size_t requested_workers,
    bool include_horizontal,
    bool include_vertical
) {
    if (sample_count < 4 || dt <= 0.0 || maximum_distance <= 0.0 || maximum_frequency <= 0.0) {
        throw std::invalid_argument("invalid f-k sampling parameters");
    }
    if (sample_count % 2 != 0) --sample_count;
    const double maximum_vp = *std::max_element(vp.begin(), vp.end());
    const double minimum_vs = *std::min_element(vs.begin(), vs.end());
    const std::size_t period_count = static_cast<std::size_t>(
        std::ceil(sample_count * dt * maximum_vp / maximum_distance)
    ) + 1;
    const double spatial_period = period_count * maximum_distance;
    const double frequency_step = 1.0 / (sample_count * dt);
    const double nyquist = 0.5 / dt;
    maximum_frequency = std::min(maximum_frequency, nyquist);
    const std::size_t frequency_count = static_cast<std::size_t>(
        std::ceil(maximum_frequency / frequency_step)
    ) + 1;
    const double omega_maximum = 2.0 * M_PI * maximum_frequency;
    const double kmax1 = 2.0 * omega_maximum / minimum_vs;
    double minimum_finite_thickness = std::numeric_limits<double>::infinity();
    for (double layer_thickness : thickness) {
        if (std::isfinite(layer_thickness) && layer_thickness > 0.0) {
            minimum_finite_thickness = std::min(minimum_finite_thickness, layer_thickness);
        }
    }
    const double layer_term = std::isfinite(minimum_finite_thickness)
        ? 1.5 * M_PI / minimum_finite_thickness : 0.0;
    const double kmax2 = std::sqrt(
        layer_term * layer_term + std::pow(omega_maximum / minimum_vs, 2.0)
    );
    const double maximum_wavenumber = std::sqrt(2.0) * std::max(kmax1, kmax2);
    const double wavenumber_step = 2.0 * M_PI / spatial_period;
    std::size_t wavenumber_intervals = static_cast<std::size_t>(
        std::ceil(maximum_wavenumber / wavenumber_step)
    );
    if (wavenumber_intervals % 2 != 0) ++wavenumber_intervals;
    const std::size_t wavenumber_count = wavenumber_intervals + 1;
    const double ewm_shift = 2.0 * M_PI * frequency_step;

    VerticalFkResponse response;
    response.frequency.resize(frequency_count);
    response.wavenumber.resize(wavenumber_count);
    const std::size_t response_size = frequency_count * wavenumber_count;
    if (include_horizontal) {
        response.hr.assign(response_size, Complex(0.0, 0.0));
        response.ht.assign(response_size, Complex(0.0, 0.0));
        response.hv.assign(response_size, Complex(0.0, 0.0));
    }
    if (include_vertical) {
        response.vr.assign(response_size, Complex(0.0, 0.0));
        response.vv.assign(response_size, Complex(0.0, 0.0));
    }
    for (std::size_t index = 0; index < frequency_count; ++index) {
        response.frequency[index] = index * frequency_step;
    }
    for (std::size_t index = 0; index < wavenumber_count; ++index) {
        response.wavenumber[index] = index * wavenumber_step;
    }

    std::atomic<std::size_t> next_frequency(1);
    auto compute_frequency = [&]() {
        while (true) {
            const std::size_t frequency_index = next_frequency.fetch_add(1);
            if (frequency_index >= frequency_count) return;
            const double angular_frequency = 2.0 * M_PI * response.frequency[frequency_index];
            const Complex load_factor = std::exp(Complex(0.0, -angular_frequency * delay))
                                      * std::exp(-ewm_shift * delay) / (2.0 * M_PI);
            const Complex shifted_omega(angular_frequency, -ewm_shift);
            for (std::size_t wavenumber_index = 0;
                 wavenumber_index < wavenumber_count; ++wavenumber_index) {
                auto [stiffness_sh, stiffness_svp] = global_stiffness(
                    vp, vs, damp_p, damp_s, density, thickness,
                    response.wavenumber[wavenumber_index], shifted_omega
                );
                const std::size_t output_index = frequency_index * wavenumber_count
                                               + wavenumber_index;
                if (include_horizontal) {
                    std::vector<Complex> radial_load(stiffness_svp.rows, Complex(0.0, 0.0));
                    radial_load[0] = load_factor;
                    const std::vector<Complex> radial_displacement = solve_linear(
                        stiffness_svp, std::move(radial_load)
                    );
                    std::vector<Complex> tangential_load(stiffness_sh.rows, Complex(0.0, 0.0));
                    tangential_load[0] = load_factor;
                    const std::vector<Complex> tangential_displacement = solve_linear(
                        std::move(stiffness_sh), std::move(tangential_load)
                    );
                    response.hr[output_index] = radial_displacement[0];
                    response.ht[output_index] = tangential_displacement[0];
                    response.hv[output_index] = radial_displacement[1];
                }
                if (include_vertical) {
                    std::vector<Complex> vertical_load(stiffness_svp.rows, Complex(0.0, 0.0));
                    vertical_load[1] = load_factor;
                    const std::vector<Complex> vertical_displacement = solve_linear(
                        std::move(stiffness_svp), std::move(vertical_load)
                    );
                    response.vr[output_index] = vertical_displacement[0];
                    response.vv[output_index] = vertical_displacement[1];
                }
            }
        }
    };
    std::size_t worker_count = requested_workers;
    if (worker_count == 0) worker_count = std::thread::hardware_concurrency();
    worker_count = std::max<std::size_t>(1, std::min(worker_count, frequency_count - 1));
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
        workers.emplace_back(compute_frequency);
    }
    for (std::thread& worker : workers) worker.join();
    return response;
}

static py::array_t<Complex> matrix_to_numpy(const Matrix& matrix) {
    py::array_t<Complex> result({matrix.rows, matrix.cols});
    auto output = result.mutable_unchecked<2>();
    for (std::size_t row = 0; row < matrix.rows; ++row) {
        for (std::size_t column = 0; column < matrix.cols; ++column) {
            output(row, column) = matrix(row, column);
        }
    }
    return result;
}

static std::vector<double> numpy_vector(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& array
) {
    if (array.ndim() != 1) {
        throw std::invalid_argument("model inputs must be one-dimensional");
    }
    auto values = array.unchecked<1>();
    std::vector<double> result(static_cast<std::size_t>(values.shape(0)));
    for (py::ssize_t index = 0; index < values.shape(0); ++index) {
        result[static_cast<std::size_t>(index)] = values(index);
    }
    return result;
}

static py::array_t<double> radial_transform_lookup(
    const py::array_t<Complex, py::array::c_style | py::array::forcecast>& spectrum,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& frequency,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& distance,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& velocity,
    const py::array_t<Complex, py::array::c_style | py::array::forcecast>& lookup_kernel,
    double lookup_minimum,
    double lookup_step,
    std::size_t requested_workers
) {
    if (spectrum.ndim() != 2 || frequency.ndim() != 1 || distance.ndim() != 1
        || velocity.ndim() != 1 || lookup_kernel.ndim() != 1) {
        throw std::invalid_argument("radial transform inputs have invalid dimensions");
    }
    const std::size_t frequency_count = static_cast<std::size_t>(frequency.shape(0));
    const std::size_t distance_count = static_cast<std::size_t>(distance.shape(0));
    const std::size_t velocity_count = static_cast<std::size_t>(velocity.shape(0));
    const std::size_t lookup_count = static_cast<std::size_t>(lookup_kernel.shape(0));
    if (static_cast<std::size_t>(spectrum.shape(0)) != frequency_count
        || static_cast<std::size_t>(spectrum.shape(1)) != distance_count
        || lookup_count < 2 || lookup_step <= 0.0) {
        throw std::invalid_argument("radial transform input sizes are inconsistent");
    }
    const Complex* spectrum_data = spectrum.data();
    const double* frequency_data = frequency.data();
    const double* distance_data = distance.data();
    const double* velocity_data = velocity.data();
    const Complex* lookup_data = lookup_kernel.data();
    std::vector<double> raw(velocity_count * frequency_count, 0.0);
    const std::size_t task_count = velocity_count * frequency_count;
    std::atomic<std::size_t> next_task(0);
    auto compute_task = [&]() {
        while (true) {
            const std::size_t task = next_task.fetch_add(1);
            if (task >= task_count) return;
            const std::size_t velocity_index = task / frequency_count;
            const std::size_t frequency_index = task % frequency_count;
            const double scale = 2.0 * M_PI * frequency_data[frequency_index]
                               / velocity_data[velocity_index];
            Complex sum(0.0, 0.0);
            const Complex* spectrum_row = spectrum_data + frequency_index * distance_count;
            for (std::size_t distance_index = 0;
                 distance_index < distance_count; ++distance_index) {
                const double argument = scale * distance_data[distance_index];
                double position = (argument - lookup_minimum) / lookup_step;
                position = std::max(0.0, std::min(
                    position, static_cast<double>(lookup_count - 1)
                ));
                std::size_t left = static_cast<std::size_t>(position);
                if (left >= lookup_count - 1) left = lookup_count - 2;
                const double fraction = position - static_cast<double>(left);
                const Complex kernel = lookup_data[left]
                                     + fraction * (lookup_data[left + 1] - lookup_data[left]);
                sum += kernel * spectrum_row[distance_index];
            }
            raw[velocity_index * frequency_count + frequency_index] = std::abs(sum);
        }
    };
    std::size_t worker_count = requested_workers;
    if (worker_count == 0) worker_count = std::thread::hardware_concurrency();
    worker_count = std::max<std::size_t>(1, std::min(worker_count, task_count));
    {
        py::gil_scoped_release release;
        std::vector<std::thread> workers;
        workers.reserve(worker_count);
        for (std::size_t worker = 0; worker < worker_count; ++worker) {
            workers.emplace_back(compute_task);
        }
        for (std::thread& worker : workers) worker.join();
    }
    py::array_t<double> output({
        static_cast<py::ssize_t>(velocity_count),
        static_cast<py::ssize_t>(frequency_count),
    });
    std::copy(raw.begin(), raw.end(), output.mutable_data());
    return output;
}

PYBIND11_MODULE(_dwm_core, module) {
    module.doc() = "WaveDispX C++ kernels for the discrete-wavenumber layered-medium solver";
    module.def(
        "stiff_element",
        [](double cp, double cs, double damp_p, double damp_s, double density,
           double thickness, double wavenumber, Complex omega) {
            auto [svp, sh] = stiff_element(
                cp, cs, damp_p, damp_s, density, thickness, wavenumber, omega
            );
            return py::make_tuple(matrix_to_numpy(svp), matrix_to_numpy(sh));
        },
        py::arg("vp_mps"), py::arg("vs_mps"), py::arg("damp_p"), py::arg("damp_s"),
        py::arg("density_kgm3"), py::arg("thickness_m"), py::arg("wavenumber_radpm"),
        py::arg("omega_radps")
    );
    module.def(
        "global_stiffness",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& vp,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& vs,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& damp_p,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& damp_s,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& density,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& thickness,
           double wavenumber, Complex omega) {
            auto [sh, svp] = global_stiffness(
                numpy_vector(vp), numpy_vector(vs), numpy_vector(damp_p),
                numpy_vector(damp_s), numpy_vector(density), numpy_vector(thickness),
                wavenumber, omega
            );
            return py::make_tuple(matrix_to_numpy(sh), matrix_to_numpy(svp));
        },
        py::arg("vp_mps"), py::arg("vs_mps"), py::arg("damp_p"), py::arg("damp_s"),
        py::arg("density_kgm3"), py::arg("thickness_m"), py::arg("wavenumber_radpm"),
        py::arg("omega_radps")
    );
    module.def(
        "radial_transform_lookup",
        &radial_transform_lookup,
        py::arg("weighted_spectrum"), py::arg("frequency_hz"),
        py::arg("distance_m"), py::arg("velocity_mps"), py::arg("lookup_kernel"),
        py::arg("lookup_minimum"), py::arg("lookup_step"), py::arg("workers") = 0
    );
    module.def(
        "fk_response",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& vp,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& vs,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& damp_p,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& damp_s,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& density,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& thickness,
           std::size_t sample_count, double dt, double maximum_distance, double delay,
           double maximum_frequency, std::size_t workers,
           bool include_horizontal, bool include_vertical) {
            const std::vector<double> vp_values = numpy_vector(vp);
            const std::vector<double> vs_values = numpy_vector(vs);
            const std::vector<double> damp_p_values = numpy_vector(damp_p);
            const std::vector<double> damp_s_values = numpy_vector(damp_s);
            const std::vector<double> density_values = numpy_vector(density);
            const std::vector<double> thickness_values = numpy_vector(thickness);
            VerticalFkResponse response;
            {
                py::gil_scoped_release release;
                response = fk_response(
                    vp_values, vs_values, damp_p_values, damp_s_values,
                    density_values, thickness_values,
                    sample_count, dt, maximum_distance, delay, maximum_frequency, workers,
                    include_horizontal, include_vertical
                );
            }
            const py::ssize_t frequency_count = static_cast<py::ssize_t>(response.frequency.size());
            const py::ssize_t wavenumber_count = static_cast<py::ssize_t>(response.wavenumber.size());
            py::array_t<double> frequency(frequency_count);
            py::array_t<double> wavenumber(wavenumber_count);
            std::copy(response.frequency.begin(), response.frequency.end(), frequency.mutable_data());
            std::copy(response.wavenumber.begin(), response.wavenumber.end(), wavenumber.mutable_data());
            py::dict result;
            result["frequency_hz"] = std::move(frequency);
            result["wavenumber_radpm"] = std::move(wavenumber);
            auto add_field = [&](const char* name, const std::vector<Complex>& values) {
                if (values.empty()) return;
                py::array_t<Complex> field({frequency_count, wavenumber_count});
                std::copy(values.begin(), values.end(), field.mutable_data());
                result[name] = std::move(field);
            };
            add_field("hr", response.hr);
            add_field("ht", response.ht);
            add_field("hv", response.hv);
            add_field("vr", response.vr);
            add_field("vv", response.vv);
            return result;
        },
        py::arg("vp_mps"), py::arg("vs_mps"), py::arg("damp_p"), py::arg("damp_s"),
        py::arg("density_kgm3"), py::arg("thickness_m"), py::arg("sample_count"),
        py::arg("dt_s"), py::arg("maximum_distance_m"), py::arg("delay_s"),
        py::arg("maximum_frequency_hz"), py::arg("workers") = 0,
        py::arg("include_horizontal") = true, py::arg("include_vertical") = true
    );
}
