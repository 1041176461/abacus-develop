#include "esolver_ks_lcao_tddft.h"

#include "module_io/cal_r_overlap_R.h"
#include "module_io/dipole_io.h"
#include "module_io/td_current_io.h"
#include "module_io/write_HS.h"
#include "module_io/write_HS_R.h"
#include "module_io/write_wfc_nao.h"
#include "module_io/output_log.h"

//--------------temporary----------------------------
#include "module_base/blas_connector.h"
#include "module_base/global_function.h"
#include "module_base/lapack_connector.h"
#include "module_base/scalapack_connector.h"
#include "module_elecstate/module_charge/symmetry_rho.h"
#include "module_elecstate/occupy.h"
#include "module_hamilt_lcao/hamilt_lcaodft/LCAO_domain.h" // need divide_HS_in_frag
#include "module_hamilt_lcao/module_tddft/evolve_elec.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"
#include "module_io/print_info.h"
#include "module_hamilt_lcao/module_deltaspin/spin_constrain.h"
#include "module_hamilt_general/module_ewald/H_Ewald_pw.h"
#include "module_elecstate/potentials/H_TDDFT_pw.h"

//-----HSolver ElecState Hamilt--------
#include "module_elecstate/elecstate_lcao.h"
#include "module_elecstate/elecstate_lcao_tddft.h"
#include "module_hamilt_lcao/hamilt_lcaodft/hamilt_lcao.h"
#include "module_hsolver/hsolver_lcao.h"
#include "module_parameter/parameter.h"
#include "module_psi/psi.h"

//-----force& stress-------------------
#include "module_hamilt_lcao/hamilt_lcaodft/FORCE_STRESS.h"

//---------------------------------------------------

namespace ModuleESolver
{

ESolver_KS_LCAO_TDDFT::ESolver_KS_LCAO_TDDFT()
{
    classname = "ESolver_KS_LCAO_TDDFT";
    basisname = "LCAO";
}

ESolver_KS_LCAO_TDDFT::~ESolver_KS_LCAO_TDDFT()
{
    delete psi_laststep;
    if (Hk_laststep != nullptr)
    {
        for (int ik = 0; ik < kv.get_nks(); ++ik)
        {
            delete[] Hk_laststep[ik];
        }
        delete[] Hk_laststep;
    }
    if (Sk_laststep != nullptr)
    {
        for (int ik = 0; ik < kv.get_nks(); ++ik)
        {
            delete[] Sk_laststep[ik];
        }
        delete[] Sk_laststep;
    }
    if (td_p != nullptr)
    {
        delete td_p;
    }
    TD_Velocity::td_vel_op = nullptr;

}

void ESolver_KS_LCAO_TDDFT::before_all_runners(const Input_para& inp, UnitCell& ucell)
{
    // 1) run "before_all_runners" in ESolver_KS
    ESolver_KS::before_all_runners(inp, ucell);

    // 2) initialize the local pseudopotential with plane wave basis set
    GlobalC::ppcell.init_vloc(GlobalC::ppcell.vloc, pw_rho);

    // 3) initialize the electronic states for TDDFT
    if (this->pelec == nullptr)
    {
        this->pelec = new elecstate::ElecStateLCAO_TDDFT(&this->chr,
                                                         &kv,
                                                         kv.get_nks(),
                                                         &this->GK, // mohan add 2024-04-01
                                                         this->pw_rho,
                                                         pw_big);
    }

    // 4) read the local orbitals and construct the interpolation tables.
    // initialize the pv
    LCAO_domain::init_basis_lcao(this->pv,
                                 inp.onsite_radius,
                                 inp.lcao_ecut,
                                 inp.lcao_dk,
                                 inp.lcao_dr,
                                 inp.lcao_rmax,
                                 ucell,
                                 two_center_bundle_,
                                 orb_);

    // 5) allocate H and S matrices according to computational resources
    LCAO_domain::divide_HS_in_frag(PARAM.globalv.gamma_only_local, this->pv, kv.get_nks(), orb_);

    // 6) initialize Density Matrix
    dynamic_cast<elecstate::ElecStateLCAO<std::complex<double>>*>(this->pelec)
        ->init_DM(&kv, &this->pv, PARAM.inp.nspin);

#ifdef __EXX
    if (GlobalC::exx_info.info_global.cal_exx)
    {
        XC_Functional::set_xc_first_loop(ucell);
        // initialize 2-center radial tables for EXX-LRI
        if (GlobalC::exx_info.info_ri.real_number)
        {
            this->exx_lri_double->init(MPI_COMM_WORLD, this->kv, orb_);
        }
        else
        {
            this->exx_lri_complex->init(MPI_COMM_WORLD, this->kv, orb_);
        }
    }
#endif

    // 8) initialize the charge density
    this->pelec->charge->allocate(PARAM.inp.nspin);
    this->pelec->omega = GlobalC::ucell.omega; // this line is very odd.

    // 9) initializee the potential
    this->pelec->pot = new elecstate::Potential(pw_rhod,
                                                pw_rho,
                                                &GlobalC::ucell,
                                                &(GlobalC::ppcell.vloc),
                                                &(sf),
                                                &(pelec->f_en.etxc),
                                                &(pelec->f_en.vtxc));

    // this line should be optimized
    this->pelec_td = dynamic_cast<elecstate::ElecStateLCAO_TDDFT*>(this->pelec);

    this->atoms_fixed = !ucell.if_atoms_can_move();

    td_p  = new TD_Velocity(&GlobalC::ucell);
    TD_Velocity::td_vel_op = td_p;
}

//------------------------------------------------------------------------------
//! the 7th function of ESolver_KS: run
//! mohan add 2024-05-11
//! 2) before_scf (electronic iteration loops)
//! 3) run charge density
//! 4) SCF iterations
//! 5) write head
//! 6) initialization of SCF iterations
//! 7) use Hamiltonian to obtain charge density
//! 8) for MPI: STOGROUP? need to rewrite
//! 9) update potential
//! 10) finish scf iterations
//! 11) get mtaGGA related parameters
//! 12) Json, need to be moved to somewhere else
//! 13) check convergence
//! 14) add Json of efermi energy converge
//! 15) after scf
//! 16) Json again
//------------------------------------------------------------------------------
void ESolver_KS_LCAO_TDDFT::runner(const int istep, UnitCell& ucell)
{
    ModuleBase::TITLE("ESolver_KS_LCAO_TDDFT", "runner");
    ModuleBase::timer::tick(this->classname, "runner");
    // done
    // 2) before_scf (electronic iteration loops)
    ModuleBase::timer::tick(this->classname, "before_scf");
    this->before_scf(istep);
    ModuleBase::timer::tick(this->classname, "before_scf");
    // things only initialize once
    this->pelec_td->first_evolve = true;
    if(!TD_Velocity::tddft_velocity && TD_Velocity::out_current)
    {
        // initialize the velocity operator
        velocity_mat = new TD_current(&GlobalC::ucell, &GlobalC::GridD, &this->pv, orb_, two_center_bundle_.overlap_orb.get());
        //calculate velocity operator
        velocity_mat->calculate_grad_term();
        velocity_mat->calculate_vcomm_r();
    }
    int estep_max = (istep == 0) ? PARAM.inp.estep_per_md +1 : PARAM.inp.estep_per_md;
    for(int estep =0; estep < estep_max; estep++)
    {
        // calculate total time step
        this->totstep++;
        this->print_step();
        //this->p_chgmix->mix_reset();
        //update At
        if(elecstate::H_TDDFT_pw::stype!=0)
        {
            elecstate::H_TDDFT_pw::update_At();
            td_p->cal_cart_At(elecstate::H_TDDFT_pw::At);
        }
        std::cout << "cart_At: " << td_p->cart_At[0] << " " << td_p->cart_At[1] << " " << td_p->cart_At[2] << std::endl;
        std::cout<<"Et: "<<elecstate::H_TDDFT_pw::Et[0]<<" "<<elecstate::H_TDDFT_pw::Et[1]<<" "<<elecstate::H_TDDFT_pw::Et[2]<<std::endl;
        if(estep!=0)
        {
            this->CE.update_all_dis(GlobalC::ucell);
            this->CE.extrapolate_charge(
#ifdef __MPI
            &(GlobalC::Pgrid),
#endif
            GlobalC::ucell,
            this->pelec->charge,
            &(this->sf),
            GlobalV::ofs_running,
            GlobalV::ofs_warning);
            //need to test if correct when estep>0
            this->pelec_td->init_scf(totstep, this->sf.strucFac, GlobalC::ucell.symm);
            dynamic_cast<elecstate::ElecStateLCAO<std::complex<double>>*>(this->pelec)->get_DM()->cal_DMR();
            if(totstep <= PARAM.inp.td_tend + 1)
            {
                TD_Velocity::evolve_once = true;
            }
        }
        ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running, "INIT SCF");
        // perhaps no need change
        this->conv_esolver = false;
        this->niter = this->maxniter;
        // no need to change
        // 4) SCF iterations
        double diag_ethr = PARAM.inp.pw_diag_thr;

        std::cout << " * * * * * *\n << Start SCF iteration." << std::endl;
        for (int iter = 1; iter <= this->maxniter; ++iter)
        {
            //no need to change
            // 5) write head
            ModuleIO::write_head_td(GlobalV::ofs_running, istep, estep, iter, this->basisname);

    #ifdef __MPI
            auto iterstart = MPI_Wtime();
    #else
            auto iterstart = std::chrono::system_clock::now();
    #endif

            // probably no need to change
            // 6) initialization of SCF iterations
            this->iter_init(totstep, iter);
            // no need to change
            // 7) use Hamiltonian to obtain charge density
            this->hamilt2density(totstep, iter, diag_ethr);

            // 8) for MPI: STOGROUP? need to rewrite
            //<Temporary> It may be changed when more clever parallel algorithm is
            // put forward.
            // When parallel algorithm for bands are adopted. Density will only be
            // treated in the first group.
            //(Different ranks should have abtained the same, but small differences
            // always exist in practice.)
            // Maybe in the future, density and wavefunctions should use different
            // parallel algorithms, in which they do not occupy all processors, for
            // example wavefunctions uses 20 processors while density uses 10.
            if (GlobalV::MY_STOGROUP == 0)
            {   // check chg between two estep 
                // double drho = this->estate.caldr2();
                // EState should be used after it is constructed.
                drho = p_chgmix->get_drho(pelec->charge, PARAM.inp.nelec);
                // no need to change
                if (PARAM.inp.scf_os_stop) // if oscillation is detected, SCF will stop
                {
                    this->oscillate_esolver = this->p_chgmix->if_scf_oscillate(iter, drho, PARAM.inp.scf_os_ndim, PARAM.inp.scf_os_thr);
                }
                // change done
                this->conv_esolver = (drho < this->scf_thr);

                // If drho < hsolver_error in the first iter or drho < scf_thr, we
                // do not change rho.
                // no need to cange
                if(!this->conv_esolver)
                {
                    //----------charge mixing---------------
                    p_chgmix->mix_rho(pelec->charge); // update chr->rho by mixing
                    if (PARAM.inp.scf_thr_type == 2)
                    {
                        pelec->charge->renormalize_rho(); // renormalize rho in R-space would
                                                        // induce a error in K-space
                    }
                    //----------charge mixing done-----------
                }
            }
    #ifdef __MPI
            MPI_Bcast(&drho, 1, MPI_DOUBLE, 0, PARAPW_WORLD);
            MPI_Bcast(&this->conv_esolver, 1, MPI_DOUBLE, 0, PARAPW_WORLD);
            MPI_Bcast(pelec->charge->rho[0], this->pw_rhod->nrxx, MPI_DOUBLE, 0, PARAPW_WORLD);
    #endif
            // no need to change
            // 9) update potential
            // Hamilt should be used after it is constructed.
            // this->phamilt->update(conv_esolver);
            this->update_pot(totstep, iter);
            // no need to change
            // 10) finish scf iterations
            this->iter_finish(totstep, iter);
    #ifdef __MPI
            double duration = (double)(MPI_Wtime() - iterstart);
    #else
            double duration
                = (std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - iterstart))
                    .count()
                / static_cast<double>(1e6);
    #endif
            // not change for now, perhaps do no harm
            // 11) get mtaGGA related parameters
            double dkin = 0.0; // for meta-GGA
            if (XC_Functional::get_func_type() == 3 || XC_Functional::get_func_type() == 5)
            {
                dkin = p_chgmix->get_dkin(pelec->charge, PARAM.inp.nelec);
            }
            this->pelec->print_etot(this->conv_esolver, iter, drho, dkin, duration, PARAM.inp.printe, diag_ethr);

            // 12) Json, need to be moved to somewhere else
    #ifdef __RAPIDJSON
            // add Json of scf mag
            Json::add_output_scf_mag(GlobalC::ucell.magnet.tot_magnetization,
                                    GlobalC::ucell.magnet.abs_magnetization,
                                    this->pelec->f_en.etot * ModuleBase::Ry_to_eV,
                                    this->pelec->f_en.etot_delta * ModuleBase::Ry_to_eV,
                                    drho,
                                    duration);
    #endif //__RAPIDJSON
            // no need to change
            // 13) check convergence
            if (this->conv_esolver || this->oscillate_esolver)
            {
                this->niter = iter;
                if (this->oscillate_esolver)
                {
                    std::cout << " !! Density oscillation is found, STOP HERE !!" << std::endl;
                }
                break;
            }

        } // end scf iterations
        std::cout << " >> Leave SCF iteration.\n * * * * * *" << std::endl;
        // change done
        // 15) after scf
        ModuleBase::timer::tick(this->classname, "after_scf");
        this->after_scf(totstep);
        ModuleBase::timer::tick(this->classname, "after_scf");
        this->pelec_td->first_evolve = false;
        if(!restart_done)
        {
            estep += PARAM.inp.estep_shift%PARAM.inp.estep_per_md;
            totstep += PARAM.inp.estep_shift;
            restart_done = true;
        }
    }
    if(!TD_Velocity::tddft_velocity && TD_Velocity::out_current)
    {
        delete velocity_mat;
    }
    ModuleBase::timer::tick(this->classname, "runner");
    return;
};
void ESolver_KS_LCAO_TDDFT::print_step()
{
    std::cout << " -------------------------------------------" << std::endl;
	GlobalV::ofs_running << "\n -------------------------------------------" << std::endl;
    std::cout << " STEP OF ELECTRON EVOLVE : " << unsigned(totstep) << std::endl;
	GlobalV::ofs_running << " STEP OF ELECTRON EVOLVE : " << unsigned(totstep) << std::endl;
    std::cout << " -------------------------------------------" << std::endl;
    GlobalV::ofs_running << " -------------------------------------------" << std::endl;
}

void ESolver_KS_LCAO_TDDFT::iter_init(const int istep, const int iter)
{
    ModuleBase::TITLE("ESolver_KS_LCAO_TDDFT", "iter_init");
    if (iter == 1)
    {
        this->p_chgmix->mix_reset(); // init mixing
        this->p_chgmix->mixing_restart_step = PARAM.inp.scf_nmax + 1;
        this->p_chgmix->mixing_restart_count = 0;
    }

    // mohan update 2012-06-05
    this->pelec->f_en.deband_harris = this->pelec->cal_delta_eband();

    // mohan move it outside 2011-01-13
    // first need to calculate the weight according to
    // electrons number.
    if (istep == 0 && this->wf.init_wfc == "file")
    {
        if (iter == 1)
        {
            std::cout << " WAVEFUN -> CHARGE " << std::endl;

            // calculate the density matrix using read in wave functions
            // and the ncalculate the charge density on grid.

            this->pelec->skip_weights = true;
            this->pelec->psiToRho(*this->psi);
            this->pelec->skip_weights = false;

            // calculate the local potential(rho) again.
            // the grid integration will do in later grid integration.

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            // a puzzle remains here.
            // if I don't renew potential,
            // The scf_thr is very small.
            // OneElectron, Hartree and
            // Exc energy are all correct
            // except the band energy.
            //
            // solved by mohan 2010-09-10
            // there are there rho here:
            // rho1: formed by read in orbitals.
            // rho2: atomic rho, used to construct H
            // rho3: generated by after diagonalize
            // here converged because rho3 and rho1
            // are very close.
            // so be careful here, make sure
            // rho1 and rho2 are the same rho.
            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            if (PARAM.inp.nspin == 4)
            {
                GlobalC::ucell.cal_ux();
            }

            //! update the potentials by using new electron charge density
            this->pelec->pot->update_from_charge(this->pelec->charge, &GlobalC::ucell);

            //! compute the correction energy for metals
            this->pelec->f_en.descf = this->pelec->cal_delta_escf();
        }
    }

#ifdef __EXX
    // calculate exact-exchange
    if (GlobalC::exx_info.info_ri.real_number)
    {
        this->exd->exx_eachiterinit(istep,
                                    *dynamic_cast<const elecstate::ElecStateLCAO<std::complex<double>>*>(this->pelec)->get_DM(),
                                    this->kv,
                                    iter);
    }
    else
    {
        this->exc->exx_eachiterinit(istep,
                                    *dynamic_cast<const elecstate::ElecStateLCAO<std::complex<double>>*>(this->pelec)->get_DM(),
                                    this->kv,
                                    iter);
    }
#endif

#ifdef __DEEPKS
    // the density matrixes of DeePKS have been updated in each iter
    GlobalC::ld.set_hr_cal(true);

    // HR in HamiltLCAO should be recalculate
    if (PARAM.inp.deepks_scf)
    {
        this->p_hamilt->refresh();
    }
#endif

    if (PARAM.inp.vl_in_h)
    {
        // update Gint_K
        if (!PARAM.globalv.gamma_only_local)
        {
            this->GK.renew();
        }
        // update real space Hamiltonian
        this->p_hamilt->refresh();
    }

    // run the inner lambda loop to contrain atomic moments with the DeltaSpin
    // method
    if (PARAM.inp.sc_mag_switch && iter > PARAM.inp.sc_scf_nmin)
    {
        SpinConstrain<std::complex<double>, base_device::DEVICE_CPU>& sc = SpinConstrain<std::complex<double>, base_device::DEVICE_CPU>::getScInstance();
        sc.run_lambda_loop(iter - 1);
    }
}
void ESolver_KS_LCAO_TDDFT::cal_force(ModuleBase::matrix& force)
{
    if(atoms_fixed)
    {
        return;
    }
    ModuleBase::TITLE("ESolver_KS_LCAO_TDDFT", "cal_force");
    ModuleBase::timer::tick("ESolver_KS_LCAO_TDDFT", "cal_force");

    Force_Stress_LCAO<std::complex<double>> fsl(this->RA, GlobalC::ucell.nat);

    fsl.getForceStress(PARAM.inp.cal_force,
                       PARAM.inp.cal_stress,
                       PARAM.inp.test_force,
                       PARAM.inp.test_stress,
                       this->pv,
                       this->pelec,
                       this->psi,
                       this->GG, // mohan add 2024-04-01
                       this->GK, // mohan add 2024-04-01
                       two_center_bundle_,
                       orb_,
                       force,
                       this->scs,
                       this->sf,
                       this->kv,
                       this->pw_rho,
#ifdef __EXX
                       *this->exx_lri_double,
                       *this->exx_lri_complex,
#endif
                       &GlobalC::ucell.symm);

    // delete RA after cal_force

    this->RA.delete_grid();

    this->have_force = true;

    ModuleBase::timer::tick("ESolver_KS_LCAO_TDDFT", "cal_force");
}

void ESolver_KS_LCAO_TDDFT::hamilt2density(const int istep, const int iter, const double ethr)
{
    pelec->charge->save_rho_before_sum_band();

    if (wf.init_wfc == "file")
    {
        if (istep >= 1)
        {
            module_tddft::Evolve_elec::solve_psi(istep,
                                                 PARAM.inp.nbands,
                                                 PARAM.globalv.nlocal,
                                                 this->p_hamilt,
                                                 this->pv,
                                                 this->psi,
                                                 this->psi_laststep,
                                                 this->Hk_laststep,
                                                 this->Sk_laststep,
                                                 this->pelec_td->ekb,
                                                 td_htype,
                                                 PARAM.inp.propagator,
                                                 kv.get_nks());
        }
        this->pelec_td->psiToRho_td(this->psi[0]);
    }
    else if (istep >= 1)
    {
        module_tddft::Evolve_elec::solve_psi(istep,
                                             PARAM.inp.nbands,
                                             PARAM.globalv.nlocal,
                                             this->p_hamilt,
                                             this->pv,
                                             this->psi,
                                             this->psi_laststep,
                                             this->Hk_laststep,
                                             this->Sk_laststep,
                                             this->pelec_td->ekb,
                                             td_htype,
                                             PARAM.inp.propagator,
                                             kv.get_nks());
        this->pelec_td->psiToRho_td(this->psi[0]);
    }
    else
    {
        // reset energy
        this->pelec->f_en.eband = 0.0;
        this->pelec->f_en.demet = 0.0;
        if (this->psi != nullptr)
        {
            hsolver::HSolverLCAO<std::complex<double>> hsolver_lcao_obj(&this->pv, PARAM.inp.ks_solver);
            hsolver_lcao_obj.solve(this->p_hamilt, this->psi[0], this->pelec_td, false);
        }
    }
    // else
    // {
    //     ModuleBase::WARNING_QUIT("ESolver_KS_LCAO", "HSolver has not been initialed!");
    // }

    // print occupation of each band
    /*if (iter == 1 && istep <= 2)
    {
        GlobalV::ofs_running << "---------------------------------------------------------------"
                                "---------------------------------"
                             << std::endl;
        GlobalV::ofs_running << "occupation : " << std::endl;
        GlobalV::ofs_running << "ik  iband     occ " << std::endl;
        GlobalV::ofs_running << std::setprecision(6);
        GlobalV::ofs_running << std::setiosflags(std::ios::showpoint);
        for (int ik = 0; ik < kv.get_nks(); ik++)
        {
            for (int ib = 0; ib < PARAM.inp.nbands; ib++)
            {
                std::setprecision(6);
                GlobalV::ofs_running << ik + 1 << "     " << ib + 1 << "      " << this->pelec_td->wg(ik, ib)
                                     << std::endl;
            }
        }
        GlobalV::ofs_running << std::endl;
        GlobalV::ofs_running << "---------------------------------------------------------------"
                                "---------------------------------"
                             << std::endl;
    }*/

    for (int ik = 0; ik < kv.get_nks(); ++ik)
    {
        this->pelec_td->print_band(ik, PARAM.inp.printe, iter);
    }

#ifdef __EXX
    if (GlobalC::exx_info.info_ri.real_number)
        this->exd->exx_hamilt2density(*this->pelec, this->pv, iter);
    else
        this->exc->exx_hamilt2density(*this->pelec, this->pv, iter);
#endif

    // using new charge density.
    this->pelec->cal_energies(1);

    // symmetrize the charge density only for ground state
    if (istep <= 1)
    {
        Symmetry_rho srho;
        for (int is = 0; is < PARAM.inp.nspin; is++)
        {
            srho.begin(is, *(pelec->charge), pw_rho, GlobalC::ucell.symm);
        }
    }

    // (6) compute magnetization, only for spin==2
    GlobalC::ucell.magnet.compute_magnetization(this->pelec->charge->nrxx,
                                                this->pelec->charge->nxyz,
                                                this->pelec->charge->rho,
                                                pelec->nelec_spin.data());

    // (7) calculate delta energy
    this->pelec->f_en.deband = this->pelec->cal_delta_eband();
}

void ESolver_KS_LCAO_TDDFT::update_pot(const int istep, const int iter)
{
    // print Hamiltonian and Overlap matrix
    if (this->conv_esolver)
    {
        if (!PARAM.globalv.gamma_only_local)
        {
            this->GK.renew(true);
        }
        for (int ik = 0; ik < kv.get_nks(); ++ik)
        {
            if (PARAM.inp.out_mat_hs[0])
            {
                this->p_hamilt->updateHk(ik);
            }
            bool bit = false; // LiuXh, 2017-03-21
            // if set bit = true, there would be error in soc-multi-core
            // calculation, noted by zhengdy-soc
            if (this->psi != nullptr && (istep % PARAM.inp.out_interval == 0))
            {
                hamilt::MatrixBlock<complex<double>> h_mat, s_mat;
                this->p_hamilt->matrix(h_mat, s_mat);
                if (PARAM.inp.out_mat_hs[0])
                {
                    ModuleIO::save_mat(istep,
                                       h_mat.p,
                                       PARAM.globalv.nlocal,
                                       bit,
                                       PARAM.inp.out_mat_hs[1],
                                       1,
                                       PARAM.inp.out_app_flag,
                                       "H",
                                       "data-" + std::to_string(ik),
                                       this->pv,
                                       GlobalV::DRANK);

                    ModuleIO::save_mat(istep,
                                       s_mat.p,
                                       PARAM.globalv.nlocal,
                                       bit,
                                       PARAM.inp.out_mat_hs[1],
                                       1,
                                       PARAM.inp.out_app_flag,
                                       "S",
                                       "data-" + std::to_string(ik),
                                       this->pv,
                                       GlobalV::DRANK);
                }
            }
        }
    }

    if (elecstate::ElecStateLCAO<std::complex<double>>::out_wfc_lcao
        && (this->conv_esolver || iter == PARAM.inp.scf_nmax) && (istep % PARAM.inp.out_interval == 0) )
    {
        ModuleIO::write_wfc_nao(elecstate::ElecStateLCAO<std::complex<double>>::out_wfc_lcao,
                                this->psi[0],
                                this->pelec->ekb,
                                this->pelec->wg,
                                this->pelec->klist->kvec_c,
                                this->pv,
                                istep);
    }

    // Calculate new potential according to new Charge Density
    if (!this->conv_esolver)
    {
        if (PARAM.inp.nspin == 4)
        {
            GlobalC::ucell.cal_ux();
        }
        this->pelec->pot->update_from_charge(this->pelec->charge, &GlobalC::ucell);
        this->pelec->f_en.descf = this->pelec->cal_delta_escf();
    }
    else
    {
        this->pelec->cal_converged();
    }

    const int nloc = this->pv.nloc;
    const int ncol_nbands = this->pv.ncol_bands;
    const int nrow = this->pv.nrow;
    const int nbands = PARAM.inp.nbands;
    const int nlocal = PARAM.globalv.nlocal;

    // store wfc and Hk laststep
    if (this->conv_esolver)
    {
        if (this->psi_laststep == nullptr)
        {
#ifdef __MPI
            this->psi_laststep = new psi::Psi<std::complex<double>>(kv.get_nks(), ncol_nbands, nrow, nullptr);
#else
            this->psi_laststep = new psi::Psi<std::complex<double>>(kv.get_nks(), nbands, nlocal, nullptr);
#endif
        }

        if (td_htype == 1)
        {
            if (this->Hk_laststep == nullptr)
            {
                this->Hk_laststep = new std::complex<double>*[kv.get_nks()];
                for (int ik = 0; ik < kv.get_nks(); ++ik)
                {
                    this->Hk_laststep[ik] = new std::complex<double>[nloc];
                    ModuleBase::GlobalFunc::ZEROS(Hk_laststep[ik], nloc);
                }
            }
            if (this->Sk_laststep == nullptr)
            {
                this->Sk_laststep = new std::complex<double>*[kv.get_nks()];
                for (int ik = 0; ik < kv.get_nks(); ++ik)
                {
                    this->Sk_laststep[ik] = new std::complex<double>[nloc];
                    ModuleBase::GlobalFunc::ZEROS(Sk_laststep[ik], nloc);
                }
            }
        }

        for (int ik = 0; ik < kv.get_nks(); ++ik)
        {
            this->psi->fix_k(ik);
            this->psi_laststep->fix_k(ik);
            int size0 = psi->get_nbands() * psi->get_nbasis();
            for (int index = 0; index < size0; ++index)
            {
                psi_laststep[0].get_pointer()[index] = psi[0].get_pointer()[index];
            }

            // store Hamiltonian
            if (td_htype == 1)
            {
                this->p_hamilt->updateHk(ik);
                hamilt::MatrixBlock<complex<double>> h_mat, s_mat;
                this->p_hamilt->matrix(h_mat, s_mat);
                BlasConnector::copy(nloc, h_mat.p, 1, Hk_laststep[ik], 1);
                BlasConnector::copy(nloc, s_mat.p, 1, Sk_laststep[ik], 1);
            }
        }

            // calculate energy density matrix for tddft
            if (istep >= (wf.init_wfc == "file" ? 0 : 1) && module_tddft::Evolve_elec::td_edm == 0 && (istep+1)%PARAM.inp.estep_per_md == 0)
            {
                this->cal_edm_tddft();
            }
        }    

    // print "eigen value" for tddft
    /*if (this->conv_esolver)
    {
        GlobalV::ofs_running << "---------------------------------------------------------------"
                                "---------------------------------"
                             << std::endl;
        GlobalV::ofs_running << "Eii : " << std::endl;
        GlobalV::ofs_running << "ik  iband    Eii (eV)" << std::endl;
        GlobalV::ofs_running << std::setprecision(6);
        GlobalV::ofs_running << std::setiosflags(std::ios::showpoint);

        for (int ik = 0; ik < kv.get_nks(); ik++)
        {
            for (int ib = 0; ib < PARAM.inp.nbands; ib++)
            {
                GlobalV::ofs_running << ik + 1 << "     " << ib + 1 << "      "
                                     << this->pelec_td->ekb(ik, ib) * ModuleBase::Ry_to_eV << std::endl;
            }
        }
        GlobalV::ofs_running << std::endl;
        GlobalV::ofs_running << "---------------------------------------------------------------"
                                "---------------------------------"
                             << std::endl;
    }*/
}

void ESolver_KS_LCAO_TDDFT::after_scf(const int istep)
{
    ESolver_KS_LCAO<std::complex<double>, double>::after_scf(istep);

    for (int is = 0; is < PARAM.inp.nspin; is++)
    {
        if (module_tddft::Evolve_elec::out_dipole == 1)
        {
            std::stringstream ss_dipole;
            ss_dipole << PARAM.globalv.global_out_dir << "SPIN" << is + 1 << "_DIPOLE";
            ModuleIO::write_dipole(pelec->charge->rho_save[is], pelec->charge->rhopw, is, istep, ss_dipole.str());
        }
    }
    if (TD_Velocity::out_current == true)
    {
        elecstate::DensityMatrix<std::complex<double>, double>* tmp_DM
            = dynamic_cast<elecstate::ElecStateLCAO<std::complex<double>>*>(this->pelec)->get_DM();
        if(TD_Velocity::out_current_k)
        {
            ModuleIO::write_current_eachk(istep,
                                    this->psi,
                                    pelec,
                                    kv,
                                    two_center_bundle_.overlap_orb.get(),
                                    tmp_DM->get_paraV_pointer(),
                                    orb_,
                                    this->velocity_mat,
                                    this->RA);
        }
        else
        {
            ModuleIO::write_current(istep,
                                    this->psi,
                                    pelec,
                                    kv,
                                    two_center_bundle_.overlap_orb.get(),
                                    tmp_DM->get_paraV_pointer(),
                                    orb_,
                                    this->velocity_mat,
                                    this->RA);
        }
    }
    std::cout << "Potential (Ry): " << std::setprecision(15) << this->pelec->f_en.etot <<std::endl;
}

} // namespace ModuleESolver
