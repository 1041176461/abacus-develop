#include "td_current_io.h"

#include "module_base/global_function.h"
#include "module_base/global_variable.h"
#include "module_base/libm/libm.h"
#include "module_base/parallel_reduce.h"
#include "module_base/timer.h"
#include "module_base/tool_threading.h"
#include "module_base/vector3.h"
#include "module_elecstate/module_dm/cal_dm_psi.h"
#include "module_elecstate/potentials/H_TDDFT_pw.h"
#include "module_hamilt_lcao/hamilt_lcaodft/LCAO_domain.h"
#include "module_hamilt_lcao/module_tddft/td_velocity.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"
#include "module_parameter/parameter.h"
#ifdef __EXX
#include "module_hamilt_lcao/hamilt_lcaodft/operator_lcao/op_exx_lcao.h"
#include "module_ri/Exx_LRI.h"
#endif

#ifdef __LCAO
void ModuleIO::cal_current_exx(Exx_LRI<std::complex<double>>& exx,
                               const elecstate::DensityMatrix<std::complex<double>, double>& dm,
                               const K_Vectors& kv,
                               const Parallel_Orbitals* pv,
                               std::vector<hamilt::HContainer<std::complex<double>>*>& hR)
{
    ModuleBase::TITLE("ModuleIO", "cal_current_exx");
    ModuleBase::timer::tick("ModuleIO", "cal_current_exx");

    // gamma_only and symmetry are not supported by tddft here
    constexpr std::size_t ndim = 3;
    const int& nk = kv.get_nks() / PARAM.inp.nspin;
    std::vector<std::vector<std::complex<double>>> DMk_trans_vector = dm.get_DMK_vector();
    std::vector<const std::vector<std::complex<double>>*> DMk_trans_pointer(nk);
    for (int ik = 0; ik < nk; ++ik)
    {
        DMk_trans_pointer[ik] = &DMk_trans_vector[ik];
    }
    const std::vector<std::map<int, std::map<std::pair<int, std::array<int, ndim>>, RI::Tensor<std::complex<double>>>>>
        Ds = RI_2D_Comm::split_m2D_ktoR<std::complex<double>>(kv,
                                                              DMk_trans_pointer,
                                                              *dm.get_paraV_pointer(),
                                                              PARAM.inp.nspin);
    std::vector<
        std::vector<std::map<int, std::map<std::pair<int, std::array<int, ndim>>, RI::Tensor<std::complex<double>>>>>>
        Hexxs;
    Hexxs.resize(3);
    std::complex<double> unit = std::complex<double>{0, 1};
    RI::Cell_Nearest<int, int, 3, double, 3> cell_nearest;
    const bool use_cell_nearest = (ModuleBase::Vector3<double>(std::fmod(kv.get_koffset(0), 1.0),
                                                               std::fmod(kv.get_koffset(1), 1.0),
                                                               std::fmod(kv.get_koffset(2), 1.0))
                                       .norm()
                                   < 1e-10);
    const std::array<int, 3> Rs_period = {kv.nmp[0], kv.nmp[1], kv.nmp[2]};
    if (use_cell_nearest)
    {
        // set cell_nearest
        std::map<int, std::array<double, 3>> atoms_pos;
        for (int iat = 0; iat < GlobalC::ucell.nat; ++iat)
        {
            atoms_pos[iat] = RI_Util::Vector3_to_array3(
                GlobalC::ucell.atoms[GlobalC::ucell.iat2it[iat]].tau[GlobalC::ucell.iat2ia[iat]]);
        }
        const std::array<std::array<double, 3>, 3> latvec = {RI_Util::Vector3_to_array3(GlobalC::ucell.a1),
                                                             RI_Util::Vector3_to_array3(GlobalC::ucell.a2),
                                                             RI_Util::Vector3_to_array3(GlobalC::ucell.a3)};
        cell_nearest.init(atoms_pos, latvec, Rs_period);
    }

    for (size_t i = 0; i != ndim; ++i)
    {
        Hexxs[i].resize(PARAM.inp.nspin);
        const std::vector<std::tuple<std::set<int>, std::set<int>>> judge = RI_2D_Comm::get_2D_judge(*pv);
        for (int is = 0; is < PARAM.inp.nspin; ++is)
        {
            exx.exx_lri_td[i].set_Ds(Ds[is], GlobalC::exx_info.info_ri.dm_threshold);
            exx.exx_lri_td[i].cal_Hs();
            Hexxs[i][is]
                = LRI_CV_Tools::mul2(unit,
                                     RI::Communicate_Tensors_Map_Judge::comm_map2_first(exx.mpi_comm,
                                                                                        std::move(exx.exx_lri_td[i].Hs),
                                                                                        std::get<0>(judge[is]),
                                                                                        std::get<1>(judge[is])));
            if (use_cell_nearest)
            {
                hamilt::reallocate_hcontainer(GlobalC::ucell.nat, hR[i], Rs_period, &cell_nearest);
                RI_2D_Comm::add_HexxR(is, -0.5, Hexxs[i], *pv, PARAM.globalv.npol, *hR[i], &cell_nearest);
            }
            else
            {
                hamilt::reallocate_hcontainer(GlobalC::ucell.nat, hR[i], Rs_period);
                RI_2D_Comm::add_HexxR(is, -0.5, Hexxs[i], *pv, PARAM.globalv.npol, *hR[i], nullptr);
            }
        }
    }

    ModuleBase::timer::tick("ModuleIO", "cal_current_exx");
}

void ModuleIO::cal_tmp_DM(elecstate::DensityMatrix<std::complex<double>, double>& DM_real,
                          elecstate::DensityMatrix<std::complex<double>, double>& DM_imag,
                          int nspin)
{
    ModuleBase::TITLE("ModuleIO", "cal_tmp_DM");
    ModuleBase::timer::tick("ModuleIO", "cal_tmp_DM");
    int ld_hk = DM_real.get_paraV_pointer()->nrow;
    int ld_hk2 = 2 * ld_hk;
    for (int is = 1; is <= nspin; ++is)
    {
        for (int ik = 0; ik < DM_real.get_DMK_nks() / nspin; ++ik)
        {
            cal_tmp_DM_k(DM_real, DM_imag, ik, nspin, is, false);
        }
    }
    ModuleBase::timer::tick("ModuleIO", "cal_tmp_DM");
}
void ModuleIO::write_current(const int istep,
                             const psi::Psi<std::complex<double>>* psi,
                             const elecstate::ElecState* pelec,
                             const K_Vectors& kv,
                             const TwoCenterIntegrator* intor,
                             const Parallel_Orbitals* pv,
                             const LCAO_Orbitals& orb,
                             const TD_current* cal_current,
                             Record_adj& ra,
#ifdef __EXX
                             Exx_LRI<std::complex<double>>& exx_lri
#endif
)
{

    ModuleBase::TITLE("ModuleIO", "write_current");
    ModuleBase::timer::tick("ModuleIO", "write_current");
    std::vector<hamilt::HContainer<std::complex<double>>*> current_term = {nullptr, nullptr, nullptr};
    if (!TD_Velocity::tddft_velocity)
    {
        for (int dir = 0; dir < 3; dir++)
        {
            current_term[dir] = cal_current->get_current_term_pointer(dir);
        }
    }
    else
    {
        if (TD_Velocity::td_vel_op == nullptr)
        {
            ModuleBase::WARNING_QUIT("ModuleIO::write_current", "velocity gague infos is null!");
        }
        for (int dir = 0; dir < 3; dir++)
        {
            current_term[dir] = TD_Velocity::td_vel_op->get_current_term_pointer(dir);
        }
    }

    double omega = GlobalC::ucell.omega;
    // construct a DensityMatrix object
    // Since the function cal_dm_psi do not suport DMR in complex type, I replace it with two DMR in double type. Should
    // be refactored in the future.
    const int nspin_dm = std::map<int, int>({{1, 1}, {2, 2}, {4, 1}})[PARAM.inp.nspin];
    elecstate::DensityMatrix<std::complex<double>, double> DM_real(pv, nspin_dm, kv.kvec_d, kv.get_nks() / nspin_dm);
    elecstate::DensityMatrix<std::complex<double>, double> DM_imag(pv, nspin_dm, kv.kvec_d, kv.get_nks() / nspin_dm);
    // calculate DMK
    elecstate::cal_dm_psi(DM_real.get_paraV_pointer(), pelec->wg, psi[0], DM_real);

#ifdef __EXX
    // if (GlobalC::exx_info.info_global.cal_exx)
    // cal_current_exx(exx_lri, DM_real, kv, pv, current_term);
#endif

    // init DMR
    DM_real.init_DMR(ra, &GlobalC::ucell);
    DM_imag.init_DMR(ra, &GlobalC::ucell);
    cal_tmp_DM(DM_real, DM_imag, PARAM.inp.nspin);
    DM_real.sum_DMR_spin();
    DM_imag.sum_DMR_spin();

    double current_total[3] = {0.0, 0.0, 0.0};
#ifdef _OPENMP
#pragma omp parallel
    {
        double local_current[3] = {0.0, 0.0, 0.0};
#else
    // ModuleBase::matrix& local_soverlap = soverlap;
    double* local_current = current_total;
#endif
        ModuleBase::Vector3<double> tau1, dtau, tau2;

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
        for (int iat = 0; iat < GlobalC::ucell.nat; iat++)
        {
            const int T1 = GlobalC::ucell.iat2it[iat];
            Atom* atom1 = &GlobalC::ucell.atoms[T1];
            const int I1 = GlobalC::ucell.iat2ia[iat];
            // get iat1
            int iat1 = GlobalC::ucell.itia2iat(T1, I1);
            const int start1 = GlobalC::ucell.itiaiw2iwt(T1, I1, 0);
            for (int cb = 0; cb < ra.na_each[iat]; ++cb)
            {
                const int T2 = ra.info[iat][cb][3];
                const int I2 = ra.info[iat][cb][4];

                const int start2 = GlobalC::ucell.itiaiw2iwt(T2, I2, 0);

                Atom* atom2 = &GlobalC::ucell.atoms[T2];

                // get iat2
                int iat2 = GlobalC::ucell.itia2iat(T2, I2);
                double Rx = ra.info[iat][cb][0];
                double Ry = ra.info[iat][cb][1];
                double Rz = ra.info[iat][cb][2];
                // std::cout<< "iat1: " << iat1 << " iat2: " << iat2 << " Rx: " << Rx << " Ry: " << Ry << " Rz:" << Rz
                // << std::endl;
                //   get BaseMatrix
                hamilt::BaseMatrix<double>* tmp_matrix_real
                    = DM_real.get_DMR_pointer(1)->find_matrix(iat1, iat2, Rx, Ry, Rz);
                hamilt::BaseMatrix<double>* tmp_matrix_imag
                    = DM_imag.get_DMR_pointer(1)->find_matrix(iat1, iat2, Rx, Ry, Rz);
                // refactor
                hamilt::BaseMatrix<std::complex<double>>* tmp_m_rvx
                    = current_term[0]->find_matrix(iat1, iat2, Rx, Ry, Rz);
                hamilt::BaseMatrix<std::complex<double>>* tmp_m_rvy
                    = current_term[1]->find_matrix(iat1, iat2, Rx, Ry, Rz);
                hamilt::BaseMatrix<std::complex<double>>* tmp_m_rvz
                    = current_term[2]->find_matrix(iat1, iat2, Rx, Ry, Rz);
                if (tmp_matrix_real == nullptr)
                {
                    continue;
                }
                int row_ap = pv->atom_begin_row[iat1];
                int col_ap = pv->atom_begin_col[iat2];
                // get DMR
                for (int mu = 0; mu < pv->get_row_size(iat1); ++mu)
                {
                    for (int nu = 0; nu < pv->get_col_size(iat2); ++nu)
                    {
                        double dm2d1_real = tmp_matrix_real->get_value(mu, nu);
                        double dm2d1_imag = tmp_matrix_imag->get_value(mu, nu);

                        std::complex<double> rvx = {0, 0};
                        std::complex<double> rvy = {0, 0};
                        std::complex<double> rvz = {0, 0};

                        if (tmp_m_rvx != nullptr)
                        {
                            rvx = tmp_m_rvx->get_value(mu, nu);
                            rvy = tmp_m_rvy->get_value(mu, nu);
                            rvz = tmp_m_rvz->get_value(mu, nu);
                        }
                        // std::cout<<"mu: "<< mu <<" nu: "<< nu << std::endl;
                        // std::cout<<"dm2d1_real: "<< dm2d1_real << " dm2d1_imag: "<< dm2d1_imag << std::endl;
                        std::cout << "rvz: " << rvz.real() << " " << rvz.imag() << std::endl;
                        local_current[0] -= dm2d1_real * rvx.real() - dm2d1_imag * rvx.imag();
                        local_current[1] -= dm2d1_real * rvy.real() - dm2d1_imag * rvy.imag();
                        local_current[2] -= dm2d1_real * rvz.real() - dm2d1_imag * rvz.imag();
                    } // end kk
                } // end jj
            } // end cb
        } // end iat
#ifdef _OPENMP
#pragma omp critical(cal_current_k_reduce)
        {
            for (int i = 0; i < 3; ++i)
            {
                current_total[i] += local_current[i];
            }
        }
    }
#endif
    Parallel_Reduce::reduce_all(current_total, 3);
    // write end
    if (GlobalV::MY_RANK == 0)
    {
        std::string filename = PARAM.globalv.global_out_dir + "current_total.dat";
        std::ofstream fout;
        fout.open(filename, std::ios::app);
        fout << std::setprecision(16);
        fout << std::scientific;
        fout << istep << " " << current_total[0] / omega << " " << current_total[1] / omega << " "
             << current_total[2] / omega << std::endl;
        fout.close();
    }

    ModuleBase::timer::tick("ModuleIO", "write_current");
    return;
}
void ModuleIO::cal_tmp_DM_k(elecstate::DensityMatrix<std::complex<double>, double>& DM_real,
                            elecstate::DensityMatrix<std::complex<double>, double>& DM_imag,
                            const int ik,
                            const int nspin,
                            const int is,
                            const bool reset)
{
    ModuleBase::TITLE("ModuleIO", "cal_tmp_DM_k");
    ModuleBase::timer::tick("ModuleIO", "cal_tmp_DM_k");
    int ld_hk = DM_real.get_paraV_pointer()->nrow;
    int ld_hk2 = 2 * ld_hk;
    // tmp for is
    int ik_begin = DM_real.get_DMK_nks() / nspin * (is - 1); // jump nk for spin_down if nspin==2

    hamilt::HContainer<double>* tmp_DMR_real = DM_real.get_DMR_vector()[is - 1];
    hamilt::HContainer<double>* tmp_DMR_imag = DM_imag.get_DMR_vector()[is - 1];
    if (reset)
    {
        tmp_DMR_real->set_zero();
        tmp_DMR_imag->set_zero();
    }
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < tmp_DMR_real->size_atom_pairs(); ++i)
    {
        hamilt::AtomPair<double>& tmp_ap_real = tmp_DMR_real->get_atom_pair(i);
        hamilt::AtomPair<double>& tmp_ap_imag = tmp_DMR_imag->get_atom_pair(i);
        int iat1 = tmp_ap_real.get_atom_i();
        int iat2 = tmp_ap_real.get_atom_j();
        // get global indexes of whole matrix for each atom in this process
        int row_ap = DM_real.get_paraV_pointer()->atom_begin_row[iat1];
        int col_ap = DM_real.get_paraV_pointer()->atom_begin_col[iat2];
        for (int ir = 0; ir < tmp_ap_real.get_R_size(); ++ir)
        {
            const ModuleBase::Vector3<int> r_index = tmp_ap_real.get_R_index(ir);
            hamilt::BaseMatrix<double>* tmp_matrix_real = tmp_ap_real.find_matrix(r_index);
            hamilt::BaseMatrix<double>* tmp_matrix_imag = tmp_ap_imag.find_matrix(r_index);
#ifdef __DEBUG
            if (tmp_matrix_real == nullptr)
            {
                std::cout << "tmp_matrix is nullptr" << std::endl;
                continue;
            }
#endif
            // only ik
            if (PARAM.inp.nspin != 4)
            {
                double arg_td = 0.0;
                if (elecstate::H_TDDFT_pw::stype == 2)
                {
                    // new
                    // cal tddft phase for mixing gague
                    const int iat1 = tmp_ap_real.get_atom_i();
                    const int iat2 = tmp_ap_real.get_atom_j();
                    ModuleBase::Vector3<double> dtau
                        = TD_Velocity::td_vel_op->get_ucell()->cal_dtau(iat1, iat2, r_index);
                    double& tmp_lat0 = TD_Velocity::td_vel_op->get_ucell()->lat0;
                    arg_td = TD_Velocity::td_vel_op->cart_At * dtau * tmp_lat0;

                    /*std::cout << "arg_td " << arg_td << std::endl;
                    std::cout << "cart_At " << TD_Velocity::td_vel_op->cart_At[0] << " "<<
                    TD_Velocity::td_vel_op->cart_At[1] << " " << TD_Velocity::td_vel_op->cart_At[2] << std::endl;
                    std::cout << "dtau " << dtau[0] << " "<< dtau[1] << " " << dtau[2] << std::endl;
                    std::cout << "ucell->lat0 " << tmp_lat0 << std::endl;
                    std::cout << "iat1 " << iat1 << " " << "iat2 " << iat2 << std::endl;*/
                    // new
                }
                // cal k_phase
                // if TK==std::complex<double>, kphase is e^{ikR}
                const ModuleBase::Vector3<double> dR(r_index.x, r_index.y, r_index.z);
                const double arg = (DM_real.get_kvec_d()[ik] * dR) * ModuleBase::TWO_PI + arg_td;
                double sinp, cosp;
                ModuleBase::libm::sincos(arg, &sinp, &cosp);
                std::complex<double> kphase = std::complex<double>(cosp, sinp);
                // set DMR element
                double* tmp_DMR_real_pointer = tmp_matrix_real->get_pointer();
                double* tmp_DMR_imag_pointer = tmp_matrix_imag->get_pointer();
                std::complex<double>* tmp_DMK_pointer = DM_real.get_DMK_pointer(ik + ik_begin);
                double* DMK_real_pointer = nullptr;
                double* DMK_imag_pointer = nullptr;
                // jump DMK to fill DMR
                // DMR is row-major, DMK is column-major
                tmp_DMK_pointer += col_ap * DM_real.get_paraV_pointer()->nrow + row_ap;
                for (int mu = 0; mu < DM_real.get_paraV_pointer()->get_row_size(iat1); ++mu)
                {
                    DMK_real_pointer = (double*)tmp_DMK_pointer;
                    DMK_imag_pointer = DMK_real_pointer + 1;
                    // calculate real part
                    BlasConnector::axpy(DM_real.get_paraV_pointer()->get_col_size(iat2),
                                        -kphase.imag(),
                                        DMK_imag_pointer,
                                        ld_hk2,
                                        tmp_DMR_real_pointer,
                                        1);
                    BlasConnector::axpy(DM_real.get_paraV_pointer()->get_col_size(iat2),
                                        kphase.real(),
                                        DMK_real_pointer,
                                        ld_hk2,
                                        tmp_DMR_real_pointer,
                                        1);
                    // calculate imag part
                    BlasConnector::axpy(DM_imag.get_paraV_pointer()->get_col_size(iat2),
                                        kphase.imag(),
                                        DMK_real_pointer,
                                        ld_hk2,
                                        tmp_DMR_imag_pointer,
                                        1);
                    BlasConnector::axpy(DM_imag.get_paraV_pointer()->get_col_size(iat2),
                                        kphase.real(),
                                        DMK_imag_pointer,
                                        ld_hk2,
                                        tmp_DMR_imag_pointer,
                                        1);
                    tmp_DMK_pointer += 1;
                    tmp_DMR_real_pointer += DM_real.get_paraV_pointer()->get_col_size(iat2);
                    tmp_DMR_imag_pointer += DM_imag.get_paraV_pointer()->get_col_size(iat2);
                }
            }
        }
    }
    ModuleBase::timer::tick("ModuleIO", "cal_tmp_DM_k");
}

void ModuleIO::write_current_eachk(const int istep,
                                   const psi::Psi<std::complex<double>>* psi,
                                   const elecstate::ElecState* pelec,
                                   const K_Vectors& kv,
                                   const TwoCenterIntegrator* intor,
                                   const Parallel_Orbitals* pv,
                                   const LCAO_Orbitals& orb,
                                   const TD_current* cal_current,
                                   Record_adj& ra,
#ifdef __EXX
                                   Exx_LRI<std::complex<double>>& exx_lri
#endif
)
{

    ModuleBase::TITLE("ModuleIO", "write_current");
    ModuleBase::timer::tick("ModuleIO", "write_current");
    std::vector<hamilt::HContainer<std::complex<double>>*> current_term = {nullptr, nullptr, nullptr};
    if (!TD_Velocity::tddft_velocity)
    {
        for (int dir = 0; dir < 3; dir++)
        {
            current_term[dir] = cal_current->get_current_term_pointer(dir);
        }
    }
    else
    {
        if (TD_Velocity::td_vel_op == nullptr)
        {
            ModuleBase::WARNING_QUIT("ModuleIO::write_current", "velocity gague infos is null!");
        }
        for (int dir = 0; dir < 3; dir++)
        {
            current_term[dir] = TD_Velocity::td_vel_op->get_current_term_pointer(dir);
        }
    }
    double omega = GlobalC::ucell.omega;
    // construct a DensityMatrix object
    // Since the function cal_dm_psi do not suport DMR in complex type, I replace it with two DMR in double type. Should
    // be refactored in the future.
    const int nspin_dm = std::map<int, int>({{1, 1}, {2, 2}, {4, 1}})[PARAM.inp.nspin];
    elecstate::DensityMatrix<std::complex<double>, double> DM_real(pv, nspin_dm, kv.kvec_d, kv.get_nks() / nspin_dm);
    elecstate::DensityMatrix<std::complex<double>, double> DM_imag(pv, nspin_dm, kv.kvec_d, kv.get_nks() / nspin_dm);
    // calculate DMK
    elecstate::cal_dm_psi(DM_real.get_paraV_pointer(), pelec->wg, psi[0], DM_real);
#ifdef __EXX
    if (GlobalC::exx_info.info_global.cal_exx)
        cal_current_exx(exx_lri, DM_real, kv, pv, current_term);
#endif

    // init DMR
    DM_real.init_DMR(ra, &GlobalC::ucell);
    DM_imag.init_DMR(ra, &GlobalC::ucell);

    int nks = DM_real.get_DMK_nks();
    if (PARAM.inp.nspin == 2)
    {
        nks /= 2;
    }
    double current_total[3] = {0.0, 0.0, 0.0};
    for (int is = 1; is <= PARAM.inp.nspin; ++is)
    {
        for (int ik = 0; ik < nks; ++ik)
        {
            cal_tmp_DM_k(DM_real, DM_imag, ik, PARAM.inp.nspin, is);
            // check later
            double current_ik[3] = {0.0, 0.0, 0.0};
#ifdef _OPENMP
#pragma omp parallel
            {
                int num_threads = omp_get_num_threads();
                double local_current_ik[3] = {0.0, 0.0, 0.0};
#else
            // ModuleBase::matrix& local_soverlap = soverlap;
            double* local_current_ik = current_ik;
#endif

                ModuleBase::Vector3<double> tau1, dtau, tau2;

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
                for (int iat = 0; iat < GlobalC::ucell.nat; iat++)
                {
                    const int T1 = GlobalC::ucell.iat2it[iat];
                    Atom* atom1 = &GlobalC::ucell.atoms[T1];
                    const int I1 = GlobalC::ucell.iat2ia[iat];
                    // get iat1
                    int iat1 = GlobalC::ucell.itia2iat(T1, I1);
                    const int start1 = GlobalC::ucell.itiaiw2iwt(T1, I1, 0);
                    for (int cb = 0; cb < ra.na_each[iat]; ++cb)
                    {
                        const int T2 = ra.info[iat][cb][3];
                        const int I2 = ra.info[iat][cb][4];

                        const int start2 = GlobalC::ucell.itiaiw2iwt(T2, I2, 0);

                        Atom* atom2 = &GlobalC::ucell.atoms[T2];

                        // get iat2
                        int iat2 = GlobalC::ucell.itia2iat(T2, I2);
                        double Rx = ra.info[iat][cb][0];
                        double Ry = ra.info[iat][cb][1];
                        double Rz = ra.info[iat][cb][2];
                        // std::cout<< "iat1: " << iat1 << " iat2: " << iat2 << " Rx: " << Rx << " Ry: " << Ry << " Rz:"
                        // << Rz << std::endl;
                        //   get BaseMatrix
                        hamilt::BaseMatrix<double>* tmp_matrix_real
                            = DM_real.get_DMR_pointer(is)->find_matrix(iat1, iat2, Rx, Ry, Rz);
                        hamilt::BaseMatrix<double>* tmp_matrix_imag
                            = DM_imag.get_DMR_pointer(is)->find_matrix(iat1, iat2, Rx, Ry, Rz);
                        // refactor
                        hamilt::BaseMatrix<std::complex<double>>* tmp_m_rvx
                            = current_term[0]->find_matrix(iat1, iat2, Rx, Ry, Rz);
                        hamilt::BaseMatrix<std::complex<double>>* tmp_m_rvy
                            = current_term[1]->find_matrix(iat1, iat2, Rx, Ry, Rz);
                        hamilt::BaseMatrix<std::complex<double>>* tmp_m_rvz
                            = current_term[2]->find_matrix(iat1, iat2, Rx, Ry, Rz);
                        if (tmp_matrix_real == nullptr)
                        {
                            continue;
                        }
                        int row_ap = pv->atom_begin_row[iat1];
                        int col_ap = pv->atom_begin_col[iat2];
                        // get DMR
                        for (int mu = 0; mu < pv->get_row_size(iat1); ++mu)
                        {
                            for (int nu = 0; nu < pv->get_col_size(iat2); ++nu)
                            {
                                double dm2d1_real = tmp_matrix_real->get_value(mu, nu);
                                double dm2d1_imag = tmp_matrix_imag->get_value(mu, nu);

                                std::complex<double> rvx = {0, 0};
                                std::complex<double> rvy = {0, 0};
                                std::complex<double> rvz = {0, 0};

                                if (tmp_m_rvx != nullptr)
                                {
                                    rvx = tmp_m_rvx->get_value(mu, nu);
                                    rvy = tmp_m_rvy->get_value(mu, nu);
                                    rvz = tmp_m_rvz->get_value(mu, nu);
                                }
                                // std::cout<<"mu: "<< mu <<" nu: "<< nu << std::endl;
                                // std::cout<<"dm2d1_real: "<< dm2d1_real << " dm2d1_imag: "<< dm2d1_imag << std::endl;
                                // std::cout<<"rvz: "<< rvz.real() << " " << rvz.imag() << std::endl;
                                local_current_ik[0] -= dm2d1_real * rvx.real() - dm2d1_imag * rvx.imag();
                                local_current_ik[1] -= dm2d1_real * rvy.real() - dm2d1_imag * rvy.imag();
                                local_current_ik[2] -= dm2d1_real * rvz.real() - dm2d1_imag * rvz.imag();
                            } // end kk
                        } // end jj
                    } // end cb
                } // end iat
#ifdef _OPENMP
#pragma omp critical(cal_current_k_reduce)
                {
                    for (int i = 0; i < 3; ++i)
                    {
                        current_ik[i] += local_current_ik[i];
                    }
                }
            }
#endif
            Parallel_Reduce::reduce_all(current_ik, 3);
            for (int i = 0; i < 3; ++i)
            {
                current_total[i] += current_ik[i];
            }
            // MPI_Reduce(local_current_ik, current_ik, 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (GlobalV::MY_RANK == 0 && TD_Velocity::out_current_k)
            {
                std::string filename = PARAM.globalv.global_out_dir + "current_spin" + std::to_string(is) + "_ik"
                                       + std::to_string(ik) + ".dat";
                std::ofstream fout;
                fout.open(filename, std::ios::app);
                fout << std::setprecision(16);
                fout << std::scientific;
                fout << istep << " " << current_ik[0] / omega << " " << current_ik[1] / omega << " "
                     << current_ik[2] / omega << std::endl;
                fout.close();
            }
            // write end
        } // end nks
    } // end is
    if (GlobalV::MY_RANK == 0)
    {
        std::string filename = PARAM.globalv.global_out_dir + "current_total.dat";
        std::ofstream fout;
        fout.open(filename, std::ios::app);
        fout << std::setprecision(16);
        fout << std::scientific;
        fout << istep << " " << current_total[0] / omega << " " << current_total[1] / omega << " "
             << current_total[2] / omega << std::endl;
        fout.close();
    }

    ModuleBase::timer::tick("ModuleIO", "write_current");
    return;
}

#endif //__LCAO
