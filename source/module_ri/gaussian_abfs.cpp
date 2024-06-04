//=======================
// AUTHOR : jiyy
// DATE :   2024-02-27
//=======================

#ifndef GAUSSIAN_ABFS_CPP
#define GAUSSIAN_ABFS_CPP

#include "gaussian_abfs.h"

#include <algorithm>
// #include <chrono>

#include "module_base/global_variable.h"
#include "module_base/math_ylmreal.h"
#include "module_base/timer.h"
#include "module_base/tool_title.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"

RI::Tensor<std::complex<double>> Gaussian_Abfs::get_Vq(
    const int& lp_max,
    const int& lq_max, // Maximum L for which to calculate interaction.
    const ModuleBase::Vector3<double>& qvec,
    const ModuleBase::Matrix3& G, 
    const double& chi, // Singularity corrected value at q=0.
    const double& lambda,
    const ModuleBase::Vector3<double>& tau,
    const ORB_gaunt_table& MGT)
{
    ModuleBase::TITLE("Gaussian_Abfs", "get_Vq");
    ModuleBase::timer::tick("Gaussian_Abfs", "get_Vq");

    const int Lmax = lp_max + lq_max;
    const int n_LM = (Lmax + 1) * (Lmax + 1);
    const size_t vq_ndim0 = (lp_max + 1) * (lp_max + 1);
    const size_t vq_ndim1 = (lq_max + 1) * (lq_max + 1);
    RI::Tensor<std::complex<double>> Vq({vq_ndim0, vq_ndim1});
    /*
     n_add_ksq * 2 = lp_max + lq_max - abs(lp_max - lq_max)
        if lp_max < lq_max
            n_add_ksq * 2 = lp_max + lq_max - (lq_max - lp_max)
                          = lp_max * 2
        if lp_max > lq_max
            n_add_ksq * 2 = lp_max + lq_max - (lp_max - lq_max)
                          = lq_max * 2
        thus,
            n_add_ksq = min(lp_max, lq_max)
    */
    const int n_add_ksq = std::min(lp_max, lq_max);
    std::vector<std::vector<std::complex<double>>> lattice_sum(n_add_ksq + 1,
                                                               std::vector<std::complex<double>>(n_LM, {0.0, 0.0}));

    const double exponent = 1.0 / lambda;
    for (int i_add_ksq = 0; i_add_ksq != n_add_ksq + 1;
         ++i_add_ksq) // integrate lp, lq, L to one index i_add_ksq, i.e. (lp+lq-L)/2
    {
        const double power = -2.0 + 2 * i_add_ksq;
        const int this_Lmax = Lmax - 2 * i_add_ksq; // calculate Lmax at current lp+lq
        const bool exclude_Gamma
            = (qvec.norm() < 1e-10 && i_add_ksq == 0); // only Gamma point and lq+lp-2>0 need to be corrected
        lattice_sum[i_add_ksq] = get_lattice_sum(-qvec, G, power, exponent, exclude_Gamma, this_Lmax, tau);
    }

    /* The exponent term comes in from Taylor expanding the
        Gaussian at zero to first order in k^2, which cancels the k^-2 from the
        Coulomb interaction.  While terms of this order are in principle
        neglected, we make one exception here.  Without this, the final result
        would (slightly) depend on the Ewald lambda.*/
    if (qvec.norm() < 1e-10)
        lattice_sum[0][0] += (chi - exponent) / std::sqrt(ModuleBase::FOUR_PI);

    for (int lp = 0; lp != lp_max + 1; ++lp)
    {
        double norm_1 = double_factorial(2 * lp - 1) * std::sqrt(ModuleBase::PI * 0.5);
        for (int lq = 0; lq != lq_max + 1; ++lq)
        {
            double norm_2 = double_factorial(2 * lq - 1) * std::sqrt(ModuleBase::PI * 0.5);
            std::complex<double> phase = std::pow(ModuleBase::IMAG_UNIT, lp - lq);
            std::complex<double> cfac = ModuleBase::FOUR_PI * phase * std::pow(ModuleBase::TWO_PI, 3)
                                        / (norm_1 * norm_2) / GlobalC::ucell.omega;
            for (int L = std::abs(lp - lq); L <= lp + lq; L += 2) // if lp+lq-L == odd, then Gaunt_Coefficients = 0
            {
                const int i_add_ksq = (lp + lq - L) / 2;
                for (int mp = 0; mp != 2 * lp + 1; ++mp)
                {
                    const int lmp = MGT.get_lm_index(lp, mp);
                    for (int mq = 0; mq != 2 * lq + 1; ++mq)
                    {
                        const int lmq = MGT.get_lm_index(lq, mq);
                        for (int m = 0; m != 2 * L + 1; ++m)
                        {
                            const int lm = MGT.get_lm_index(L, m);
                            double triple_Y = MGT.Gaunt_Coefficients(lmp, lmq, lm);
                            Vq(lmp, lmq) += triple_Y * cfac * lattice_sum[i_add_ksq][lm];
                        }
                    }
                }
            }
        }
    }

    ModuleBase::timer::tick("Gaussian_Abfs", "get_Vq");
    return Vq;
}

Numerical_Orbital_Lm Gaussian_Abfs::Gauss(const Numerical_Orbital_Lm& orb, const double& lambda)
{
    Numerical_Orbital_Lm gaussian;
    const int angular_momentum_l = orb.getL();
    const double eta = 35;
    const double rcut = std::sqrt(eta / lambda);
    const double dr = orb.get_rab().back();
    const int Nr = std::ceil(rcut / dr);

    std::vector<double> rab(Nr);
    for (size_t ir = 0; ir < Nr; ++ir)
        rab[ir] = dr;
    std::vector<double> r_radial(Nr);
    for (size_t ir = 0; ir < Nr; ++ir)
        r_radial[ir] = ir * dr;

    const double frac = std::pow(lambda, angular_momentum_l + 1.5) / double_factorial(2 * angular_momentum_l - 1)
                        / std::sqrt(ModuleBase::PI * 0.5);

    std::vector<double> psi(Nr);

    for (size_t ir = 0; ir != Nr; ++ir)
        psi[ir]
            = frac * std::pow(r_radial[ir], angular_momentum_l) * std::exp(-lambda * r_radial[ir] * r_radial[ir] * 0.5);

    gaussian.set_orbital_info(orb.getLabel(),
                              orb.getType(),
                              angular_momentum_l,
                              orb.getChi(),
                              Nr,
                              ModuleBase::GlobalFunc::VECTOR_TO_PTR(rab),
                              ModuleBase::GlobalFunc::VECTOR_TO_PTR(r_radial),
                              Numerical_Orbital_Lm::Psi_Type::Psi,
                              ModuleBase::GlobalFunc::VECTOR_TO_PTR(psi),
                              orb.getNk(),
                              orb.getDk(),
                              orb.getDruniform(),
                              false,
                              true,
                              GlobalV::CAL_FORCE);

    return gaussian;
}

double Gaussian_Abfs::double_factorial(const int& n)
{
    double result = 1.0;
    for (int i = n; i > 0; i -= 2)
    {
        if (i == 1)
            result *= 1.0;
        else
            result *= static_cast<double>(i);
    }
    return result;
}

std::vector<std::complex<double>> Gaussian_Abfs::get_lattice_sum(
    const ModuleBase::Vector3<double>& qvec,
    const ModuleBase::Matrix3& G, 
    const double& power, // Will be 0. for straight GTOs and -2. for Coulomb interaction
    const double& exponent,
    const bool& exclude_Gamma, // The R==0. can be excluded by this flag.
    const int& lmax,           // Maximum angular momentum the sum is needed for.
    const ModuleBase::Vector3<double>& tau)
{
    if (power < 0.0 && !exclude_Gamma && qvec.norm() < 1e-10)
        ModuleBase::WARNING_QUIT("Gaussian_Abfs::lattice_sum", "Gamma point for power<0.0 cannot be evaluated!");

    const double eta = 35;
    const double Gmax = std::sqrt(eta / exponent) + qvec.norm() * GlobalC::ucell.tpiba;
    std::vector<int> n_supercells = get_n_supercells(G, Gmax);
    const int total_cells = std::accumulate(n_supercells.begin(), n_supercells.end(), 1, [](int a, int b) { return a * (2 * b + 1); });

    std::vector<ModuleBase::Vector3<double>> Gvec;
    Gvec.resize(3);
    Gvec[0].x = G.e11;
    Gvec[0].y = G.e12;
    Gvec[0].z = G.e13;

    Gvec[1].x = G.e21;
    Gvec[1].y = G.e22;
    Gvec[1].z = G.e23;

    Gvec[2].x = G.e31;
    Gvec[2].y = G.e32;
    Gvec[2].z = G.e33;

    const int total_lm = (lmax + 1) * (lmax + 1);
    std::vector<std::complex<double>> result(total_lm, {0.0, 0.0});
    std::vector<ModuleBase::Vector3<double>> qGvecs;
    for (int idx = 0; idx < total_cells; ++idx)
    {
        int G0 = (idx / ((2 * n_supercells[1] + 1) * (2 * n_supercells[2] + 1))) - n_supercells[0];
        int G1 = ((idx / (2 * n_supercells[2] + 1)) % (2 * n_supercells[1] + 1)) - n_supercells[1];
        int G2 = (idx % (2 * n_supercells[2] + 1)) - n_supercells[2];
        if (exclude_Gamma && G0 == 0 && G1 == 0 && G2 == 0)
            continue;
        ModuleBase::Vector3<double> qGvec = -(qvec + Gvec[0] * static_cast<double>(G0)
                                              + Gvec[1] * static_cast<double>(G1) + Gvec[2] * static_cast<double>(G2));
        qGvecs.push_back(qGvec);
    }
    const int npw = qGvecs.size();
    ModuleBase::matrix ylm(total_lm, npw);
    ModuleBase::YlmReal::Ylm_Real(total_lm, npw, qGvecs.data(), ylm);

#pragma omp declare reduction(vec_plus : std::vector<std::complex<double>> : std::transform(                           \
        omp_out.begin(),                                                                                               \
            omp_out.end(),                                                                                             \
            omp_in.begin(),                                                                                            \
            omp_out.begin(),                                                                                           \
            std::plus<std::complex<double>>())) initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))
    // auto start0 = std::chrono::system_clock::now();
#pragma omp parallel for reduction(vec_plus : result)
    for (int idx = 0; idx < npw; ++idx)
    {
        ModuleBase::Vector3<double> vec = qGvecs[idx];
        const double vec_sq = vec.norm2() * GlobalC::ucell.tpiba2;
        const double vec_abs = std::sqrt(vec_sq);

        const double val_s = std::exp(-exponent * vec_sq) * std::pow(vec_abs, power);

        std::complex<double> phase = std::exp(ModuleBase::TWO_PI * ModuleBase::IMAG_UNIT * (vec * tau));

        for (int L = 0; L != lmax + 1; ++L)
        {
            const double val_l = val_s * std::pow(vec_abs, L);
            for (int m = 0; m != 2 * L + 1; ++m)
            {
                const int lm = L * L + m;
                const double val_lm = val_l * ylm(lm, idx);
                result[lm] += val_lm * phase;
            }
        }
    }
    // auto end0 = std::chrono::system_clock::now();
    // auto duration0 = std::chrono::duration_cast<std::chrono::microseconds>(end0 - start0);
    // std::cout << "lattice Time: "
    //           << double(duration0.count()) * std::chrono::microseconds::period::num
    //                  / std::chrono::microseconds::period::den
    //           << " s" << std::endl;

    return result;
}

std::vector<int> Gaussian_Abfs::get_n_supercells(const ModuleBase::Matrix3& G, const double& Gmax)
{
    std::vector<int> n_supercells(3);
    ModuleBase::Matrix3 GI = G.Inverse();
    ModuleBase::Matrix3 latvec = GI.Transpose();
    latvec *= GlobalC::ucell.lat0;
    std::vector<ModuleBase::Vector3<double>> lat;
    lat.resize(3);
    lat[0].x = latvec.e11;
    lat[0].y = latvec.e12;
    lat[0].z = latvec.e13;
    lat[1].x = latvec.e21;
    lat[1].y = latvec.e22;
    lat[1].z = latvec.e23;
    lat[2].x = latvec.e31;
    lat[2].y = latvec.e32;
    lat[2].z = latvec.e33;

    n_supercells[0] = static_cast<int>(std::floor(lat[0].norm() * Gmax / ModuleBase::TWO_PI + 1e-5));
    n_supercells[1] = static_cast<int>(std::floor(lat[1].norm() * Gmax / ModuleBase::TWO_PI + 1e-5));
    n_supercells[2] = static_cast<int>(std::floor(lat[2].norm() * Gmax / ModuleBase::TWO_PI + 1e-5));

    return n_supercells;
}

#endif