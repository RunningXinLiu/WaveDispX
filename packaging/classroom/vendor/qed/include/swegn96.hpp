#ifndef SURFCPS_H_
#define SURFCPS_H_

#include <Eigen/Dense>
#include <map>
#include <string>

extern "C" {
void slegnpu_double_(double *thk, double *vs, double *rhom, int nlayer, double *t,
              double *cp, double *cg, double *disp, double *stress,
              double *t1, double *cp1, double *t2, double *cp2,
              double *dc2db, double *dc2dh, double *dc2dr,
              double *du2db, double *du2dh, double *du2dr, int iflsph);

void sregnpu_double_(double *thk, double *vp, double *vs, double *rhom, int nlayer,
              double *t, double *cp, double *cg, double *dispu, double *dispw,
              double *stressu, double *stressw, double *t1, double *cp1,
              double *t2, double *cp2, double *dc2da, double *dc2db,
              double *dc2dh, double *dc2dr, double *du2da, double *du2db,
              double *du2dh, double *du2dr, int iflsph);

/**
 * compute phase velocity sensitivity kernel, and group velocity
 * for Rayleigh Wave,with layer-based model.
 * note that if in spherical model, input cp should be cp_flat
 * @param nlayer no. of layers
 * @param thk,vp,vs,rhom model, shape(nlayer)
 * @param t,cp,cg period and phase/group velocity
 * @param iflsph =0 flat earth   =1 spherical earth
 * @param dispu/w stressu/w eigen function for u/w direction shape(nlayer)
 * @param dc2da(b,h,r) sensitivity kernel for vp,vs,thick and rho shape(nlayer)
 *
 */
void sregn96_(float *thk, float *vp, float *vs, float *rhom, int nlayer,
              double *t, double *cp, double *cg, double *dispu, double *dispw,
              double *stressu, double *stressw, double *dc2da, double *dc2db,
              double *dc2dh, double *dc2dr, int iflsph);

/**
 * compute phase velocity sensitivity kernel, and group velocity
 * for Love Wave,with layer-based model
 * note that if in spherical model, input cp should be cp_flat
 * @param nlayer no. of layers
 * @param thk,vs,rhom model, shape(nlayer)
 * @param t,cp,cg period and phase/group velocity
 * @param iflsph =0 flat earth   =1 spherical earth
 * @param disp,stress eigen function for u/w direction shape(nlayer)
 * @param dc2db(h,r) phase v sensitivity kernel for vs,thick and rho
 * shape(nlayer)
 */
void slegn96_(float *thk, float *vs, float *rhom, int nlayer, double *t,
              double *cp, double *cg, double *disp, double *stress,
              double *dc2db, double *dc2dh, double *dc2dr, int iflsph);

// Double-model entry points. Keep the float-model symbols above for ABI
// compatibility; native kernels must use the same model precision as Dispersion.
void sregn96_double_(double *thk, double *vp, double *vs, double *rhom, int nlayer,
                    double *t, double *cp, double *cg, double *dispu, double *dispw,
                    double *stressu, double *stressw, double *dc2da, double *dc2db,
                    double *dc2dh, double *dc2dr, int iflsph);
void slegn96_double_(double *thk, double *vs, double *rhom, int nlayer, double *t,
                    double *cp, double *cg, double *disp, double *stress,
                    double *dc2db, double *dc2dh, double *dc2dr, int iflsph);
}

class SwEgn96 {
public:
  SwEgn96(const Eigen::Ref<const Eigen::ArrayXXd> &model, bool sh,
          bool sphere = false);
  std::map<std::string, Eigen::ArrayXd> kernel(double freq, double c);

private:
  const int nl_;
  Eigen::ArrayXd tn_, rho_, vs_, vp_;
  const bool sh_;
  const bool sphere_;
};

#endif
