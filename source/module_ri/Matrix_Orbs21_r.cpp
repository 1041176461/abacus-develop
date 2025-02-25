//=======================
// AUTHOR : Yuyang Ji
// DATE :   2025-02-23
//=======================

#include "Matrix_Orbs21_r.h"
#include "exx_abfs-construct_orbs.h"
#include "module_base/timer.h"
#include "module_base/tool_title.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"

void Matrix_Orbs21_r::init(const int mode,
                           const LCAO_Orbitals& orb,
                           const double kmesh_times,
                           const double rmax,
                           int Lmax)
{
    ModuleBase::TITLE("Matrix_Orbs21_r", "init");
    ModuleBase::timer::tick("Matrix_Orbs21_r", "init");

    int Lmax_used;

    const int ntype = orb.get_ntype();
    int lmax_orb = -1, lmax_beta = -1;
    for (int it = 0; it < ntype; it++)
    {
        lmax_orb = std::max(lmax_orb, orb.Phi[it].getLmax());
        lmax_beta = std::max(lmax_beta, GlobalC::ucell.infoNL.Beta[it].getLmax());
    }
    const double dr = orb.get_dR();
    const double dk = orb.get_dk();
    const int kmesh = orb.get_kmesh() * kmesh_times + 1;
    int Rmesh = static_cast<int>(rmax / dr) + 4;
    Rmesh += 1 - Rmesh % 2;

    Center2_Orb::init_Table_Spherical_Bessel(2,
                                             mode,
                                             Lmax_used,
                                             Lmax,
                                             GlobalC::exx_info.info_ri.abfs_Lmax,
                                             lmax_orb,
                                             lmax_beta,
                                             dr,
                                             dk,
                                             kmesh,
                                             Rmesh,
                                             psb_);

    // //=========================================
    // // (3) make Gaunt coefficients table
    // //=========================================
    // this->MGT.init_Gaunt_CH(Lmax);
    // this->MGT.init_Gaunt(Lmax);

    ModuleBase::timer::tick("Matrix_Orbs21_r", "init");
}

void Matrix_Orbs21_r::init_radial(const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& orb_A,
                                  const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& orb_B,
                                  const ORB_gaunt_table& MGT)
{
    ModuleBase::TITLE("Matrix_Orbs21_r", "init_radial");
    ModuleBase::timer::tick("Matrix_Orbs21_r", "init_radial");
    Numerical_Orbital_Lm orb_r = Exx_Abfs::Construct_Orbs::construct_orb_r(orb_A);
    for (size_t TA = 0; TA != orb_A.size(); ++TA)
    {
        for (size_t TB = 0; TB != orb_B.size(); ++TB)
        {
            for (int LA = 0; LA != orb_A[TA].size(); ++LA)
            {
                for (size_t NA = 0; NA != orb_A[TA][LA].size(); ++NA)
                {
                    for (int LB = 0; LB != orb_B[TB].size(); ++LB)
                    {
                        for (size_t NB = 0; NB != orb_B[TB][LB].size(); ++NB)
                        {
                            this->center2_orb21_r[TA][TB][LA][NA][LB].insert(std::make_pair(
                                NB,
                                Center2_Orb::Orb21(orb_A[TA][LA][NA], orb_r, orb_B[TB][LB][NB], psb_, MGT)));
                        }
                    }
                }
            }
        }
    }
    ModuleBase::timer::tick("Matrix_Orbs21_r", "init_radial");
}

void Matrix_Orbs21_r::init_radial_table()
{
    ModuleBase::TITLE("Matrix_Orbs21_r", "init_radial_table");
    ModuleBase::timer::tick("Matrix_Orbs21_r", "init_radial_table");
    for (auto& co1: this->center2_orb21_r)
    {
        for (auto& co2: co1.second)
        {
            for (auto& co3: co2.second)
            {
                for (auto& co4: co3.second)
                {
                    for (auto& co5: co4.second)
                    {
                        for (auto& co6: co5.second)
                        {
                            co6.second.init_radial_table();
                        }
                    }
                }
            }
        }
    }
    ModuleBase::timer::tick("Matrix_Orbs21", "init_radial_table");
}