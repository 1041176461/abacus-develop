//=======================
// AUTHOR : Yuyang Ji
// DATE :   2025-02-23
//=======================

#include "Matrix_Orbs21_r.h"
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
    this->orbs = Exx_Abfs::Construct_Orbs::change_orbs(orb, kmesh_times);
    this->orb_r = Matrix_Orbs21_r::construct_orb_r(this->orbs);

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
                            this->center2_orb11[TA][TB][LA][NA][LB].insert(
                                std::make_pair(NB,
                                               Center2_Orb::Orb11(orb_A[TA][LA][NA], orb_B[TB][LB][NB], psb_, MGT)));
                            this->center2_orb21_r[TA][TB][LA][NA][LB].insert(std::make_pair(
                                NB,
                                Center2_Orb::Orb21(orb_A[TA][LA][NA], this->orb_r, orb_B[TB][LB][NB], psb_, MGT)));
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
    for (auto& co1: this->center2_orb11)
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

Numerical_Orbital_Lm Matrix_Orbs21_r::construct_orb_r(const LCAO_Orbitals& orb)
{
    ModuleBase::TITLE("Matrix_Orbs21_r", "construct_orb_r");
    ModuleBase::timer::tick("Matrix_Orbs21_r", "construct_orb_r");
    int orb_r_ntype = 0;
    int mat_Nr = orb.Phi[0].PhiLN(0, 0).getNr();
    int count_Nr = 0;
    Numerical_Orbital_Lm orb_rs;

    for (int T = 0; T < orb.get_ntype(); ++T)
    {
        count_Nr = orb.Phi[T].PhiLN(0, 0).getNr();
        if (count_Nr > mat_Nr)
        {
            mat_Nr = count_Nr;
            orb_r_ntype = T;
        }
    }

    orb_rs.set_orbital_info(this->orbs[orb_r_ntype][0][0].getLabel(),  // atom label
                                 orb_r_ntype,                               // atom type
                                 1,                                         // angular momentum L
                                 1,                                         // number of orbitals of this L , just N
                                 this->orbs[orb_r_ntype][0][0].getNr(),     // number of radial mesh
                                 this->orbs[orb_r_ntype][0][0].getRab(),    // the mesh interval in radial mesh
                                 this->orbs[orb_r_ntype][0][0].getRadial(), // radial mesh value(a.u.)
                                 Numerical_Orbital_Lm::Psi_Type::Psi,
                                 this->orbs[orb_r_ntype][0][0].getRadial(), // radial wave function
                                 this->orbs[orb_r_ntype][0][0].getNk(),
                                 this->orbs[orb_r_ntype][0][0].getDk(),
                                 this->orbs[orb_r_ntype][0][0].getDruniform(),
                                 false,
                                 true,
                                 PARAM.inp.cal_force);

    ModuleBase::timer::tick("Matrix_Orbs21_r", "construct_orb_r");
    return orb_rs
}