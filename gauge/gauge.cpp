#include "gauge.hpp"
#include "convenience_params.hpp"
#include <alps/hdf5/vector.hpp>
#include <alps/hdf5/multi_array.hpp>


// Defines the parameters
void gauge_sim::define_parameters(parameters_type & parameters) {
    // If the parameters are restored, they are already defined
    if (parameters.is_restored()) {
        return;
    }

    // Adds the parameters of the base class
    alps::mcbase::define_parameters(parameters);

    // Add convenience parameters of the model
    define_convenience_parameters(parameters)
        .description("Simulation of ")
        .define<int>("length", "linear size of the system")
        .define<std::string>("gauge_group", "specify the symmetry of gauge group")
        .define<int>("group_size", "specify the order of the group")
        .define<bool>("O3", 0, "determine symmetry of matter fields")
        .define<double>("temperature", 1., "actual temperature to simulate")
        .define<double>("J1", 0., "elements of J matrix")
        .define<double>("J3", 1., "elements of J matrix")
        .define<int>("total_sweeps", 0, "maximum number of sweeps")
        .define<int>("thermalization_sweeps", 10000, "number of sweeps for thermalization")
        .define<int>("hits_R", 1, "number of hits in each flip of R")
        .define<int>("hits_U", 1, "number of hits in each flip of U")
        .define<double>("global_gauge_prob", 0.05, "probability for global gauge update")
        .define<int>("sweep_unit", 10, "scale a sweep")
        .define<double>("spacing_E", 0.001, "spacing of normalized energy")
        .define<double>("spacing_nem", 0.001, "spacing of nematicity");

    parameters
        .define<bool>("resolve_sites", false, "do not average over sites (rank 1 only)")
        .define<bool>("bipartite", false, "use a bipartite configuration (AFM)")
        .define<bool>("symmetrized", true, "use symmetry <l_x m_y> == <m_y l_x>")
        .define<bool>("uniaxial", false, "only consider n-vector in configuration")
        .define<size_t>("rank", "rank of the order parameter tensor");
}


std::unique_ptr<config_policy> gauge_sim::config_policy_from_parameters(parameters_type const& parameters,
                                                                        bool unsymmetrize = true)
{
    // set up SVM configuration policy
    size_t rank = parameters["rank"].as<size_t>();
    if (parameters["resolve_sites"].as<bool>()) {
        if (rank != 1) {
            std::runtime_error("site-resolved config policy does not support "
                               "ranks other than 1 at the moment");
        }
        size_t n_sites = combinatorics::ipow(parameters["length"].as<size_t>(), 3);
        if (parameters["uniaxial"].as<bool>()) {
            return std::unique_ptr<config_policy>(
                new site_resolved_rank1_config_policy<element_policy::uniaxial>(n_sites));
        } else {
            return std::unique_ptr<config_policy>(
                new site_resolved_rank1_config_policy<element_policy::triad>(n_sites));
        }
    }
    using ctype = decltype(R);
    if (parameters["bipartite"].as<bool>()) {
        if (parameters["symmetrized"].as<bool>()) {
            if (parameters["uniaxial"].as<bool>()) {
                return std::unique_ptr<config_policy>(
                    new gauge_config_policy<lattice::square<element_policy::uniaxial, ctype>,
                                            symmetry_policy::symmetrized>(rank, unsymmetrize));
            } else {
                return std::unique_ptr<config_policy>(
                    new gauge_config_policy<lattice::square<element_policy::triad, ctype>,
                                            symmetry_policy::symmetrized>(rank, unsymmetrize));
            }
        } else {
            if (parameters["uniaxial"].as<bool>()) {
                return std::unique_ptr<config_policy>(
                    new gauge_config_policy<lattice::square<element_policy::uniaxial, ctype>,
                                            symmetry_policy::none>(rank, unsymmetrize));
            } else {
                return std::unique_ptr<config_policy>(
                    new gauge_config_policy<lattice::square<element_policy::triad, ctype>,
                                            symmetry_policy::none>(rank, unsymmetrize));
            }
        }
    }
    if (parameters["symmetrized"].as<bool>()) {
        if (parameters["uniaxial"].as<bool>()) {
            return std::unique_ptr<config_policy>(
                new gauge_config_policy<lattice::uniform<element_policy::uniaxial, ctype>,
                                        symmetry_policy::symmetrized>(rank, unsymmetrize));
        } else {
            return std::unique_ptr<config_policy>(
                new gauge_config_policy<lattice::uniform<element_policy::triad, ctype>,
                                        symmetry_policy::symmetrized>(rank, unsymmetrize));
        }
    } else {
        if (parameters["uniaxial"].as<bool>()) {
            return std::unique_ptr<config_policy>(
                new gauge_config_policy<lattice::uniform<element_policy::uniaxial, ctype>,
                                        symmetry_policy::none>(rank, unsymmetrize));
        } else {
            return std::unique_ptr<config_policy>(
                new gauge_config_policy<lattice::uniform<element_policy::triad, ctype>,
                                        symmetry_policy::none>(rank, unsymmetrize));
        }
    }
}


/** Initialize the parameters **/
gauge_sim::gauge_sim(parameters_type const & parms, std::size_t seed_offset)
    : alps::mcbase(parms, seed_offset)
    , L(parameters["length"])
    , L2(L*L)
    , L3(L*L*L)
    , gauge_group(parameters["gauge_group"].as<std::string>())
    , group_size(parameters["group_size"])
    , O3(parameters["O3"].as<bool>())
    , beta(1. / parameters["temperature"].as<double>())
    , J1(parameters["J1"].as<double>())
    , J3(parameters["J3"].as<double>())
    , Eg(-9*L3)
    , sweeps(0) //current sweeps
    , thermalization_sweeps(parameters["thermalization_sweeps"])
    , total_sweeps(parameters["total_sweeps"])
    , hits_R(parameters["hits_R"])
    , hits_U(parameters["hits_U"])
    , global_gauge_prob(parameters["global_gauge_prob"].as<double>())
    , sweep_unit(parameters["sweep_unit"])
    , spacing_E(parameters["spacing_E"].as<double>())
    , spacing_nem(parameters["spacing_nem"].as<double>())
    , current_energy(0)
    , rng(parameters["SEED"].as<std::size_t>() + seed_offset)
    , random_01(0., 1.)
    , random_11(-1., 1.)
    , random_site(0, L3 - 1)
    , random_U(0, group_size-1)
    , random_int_01(0, 1)

{
#ifdef DEBUGMODE
    if(!O3)
        std::cout << "SO(3)/" << gauge_group << " gauge model at K = 0" << std::endl;
    else
        std::cout << "O(3)/" << gauge_group << " gauge model at K = 0" << std::endl;
#endif

    R.resize(boost::extents[L3]);
    Ux.resize(boost::extents[L3]);
    Uy.resize(boost::extents[L3]);
    Uz.resize(boost::extents[L3]);


    flip_counter.resize(4);
    std::fill(flip_counter.begin(), flip_counter.end(), 0);

    /** Initialization **/
    /* Generate the gauge bath */
    pg.determine_symmetry(gauge_group, group_size, O3, gauge_bath);

    /* determine the choice of order parameter */
    if (gauge_group == "T")
        nematicity = [this]() {
            return nematicity_T();
        };
    else if (gauge_group == "Cinfv")
        nematicity = [this]() {
            return nematicity_Cinfv();
        };
    else if (gauge_group == "Dinfh")
        nematicity = [this]() {
            return nematicity_Dinfh();
        };
    else if (gauge_group == "Td")
        nematicity = [this]() {
            return nematicity_Td();
        };
    else if (gauge_group == "Th")
        nematicity = [this]() {
            return nematicity_Th();
        };
    else if (gauge_group == "O")
        nematicity = [this]() {
            return nematicity_Oh();
        };
    else if (gauge_group == "Oh")
        nematicity = [this]() {
            return nematicity_Oh();
        };
    else if (gauge_group == "I")
        nematicity = [this]() {
            return nematicity_Ih();
        };
    else if (gauge_group == "Ih")
        nematicity = [this]() {
            return nematicity_Ih();
        };
    else {
        std::cerr << "ERROR in the symmetry" << std::endl;
        return;
    }

    /** Initialization **/

    /* define J */
    J = Eigen::MatrixXd::Identity(3,3);
    J(0,0) = J1;
    J(1,1) = J1;
    J(2,2) = J3;

    /* Set to uniform */
    for(int i = 0; i < L3; i++)
    { R[i] = Eigen::MatrixXd::Identity(3,3);
        Ux[i] = Eigen::MatrixXd::Identity(3,3);
        Uy[i] = Eigen::MatrixXd::Identity(3,3);
        Uz[i] = Eigen::MatrixXd::Identity(3,3);
    }

    Eg = total_energy();

#ifdef DEBUGMODE
    std::cout << "Nematicity(normalized) at perfect order: \n "
              << sqrt(nematicity()) << std::endl;
#endif

    /* Random Initialization */
    for(int i = 0; i < L3; i++)
    {
        random_R(R[i]); //change R[i] to a random matrix
        Ux[i] = gauge_bath[random_U(rng)];
        Uy[i] = gauge_bath[random_U(rng)];
        Uz[i] = gauge_bath[random_U(rng)];
    }


    // Get initial energy
    current_energy = total_energy();

#ifdef DEBUGMODE
    std::cout << "Energy(normalized) at infinite temperature: \n "
              << current_energy/Eg << std::endl;

    std::cout << "Nematicity(normalized) at infinite temperature: \n "
              << sqrt(nematicity()) << std::endl;
#endif

    // Adds the measurements
    measurements
        << alps::accumulators::FullBinningAccumulator<double>("Energy") //normalized
        //<< alps::accumulators::FullBinningAccumulator<double>("Energy^2")
        //<< alps::accumulators::FullBinningAccumulator<double>("Energy^3")
        << alps::accumulators::FullBinningAccumulator<double>("Nematicity")
        //<< alps::accumulators::FullBinningAccumulator<double>("Nematicity^2")
        //<< alps::accumulators::FullBinningAccumulator<double>("Nematicity^4")
        ;

    confpol = config_policy_from_parameters(parameters);
}

/** Define member functions **/

void gauge_sim::flip_ratio(double current_beta, int count)
{
    /*The percentage is considered by the 0.01 factor below.*/
    double trying_R = 0.01 * count * sweep_unit * hits_R * L3;
    double trying_U = 0.01 * count * sweep_unit * hits_U * L3;

    std::vector<double> flip_ratio(flip_counter.size());
    flip_ratio[0] = flip_counter[0]/trying_R;
    flip_ratio[1] = flip_counter[1]/trying_U;
    flip_ratio[2] = flip_counter[2]/trying_U;
    flip_ratio[3] = flip_counter[3]/trying_U;


}


void gauge_sim::update()
{
    if (random_01(rng) < global_gauge_prob) {
        global_gauge_update();
    } else {
        for (int i = 0; i < L3 * sweep_unit; i++)
        {
            flip_R(random_site(rng));
            flip_Ux(random_site(rng));
            flip_Uy(random_site(rng));
            flip_Uz(random_site(rng));
        }
    }
    sweeps++;
}

void gauge_sim::global_gauge_update()
{
    Rt3 Omega;
    random_R(Omega);
    for (int i = 0; i < L3; ++i) {
        R[i] *= Omega;
    }
}

/* updates of R */
void gauge_sim::flip_R(int i)
{
    /* Find neighbours */
    int xp = i % L == 0 ? i - 1 + L : i - 1;
    int xn = (i + 1) % L == 0 ? i + 1 - L : i + 1;

    int yp = i % L2 < L ? i - L + L2 : i - L;
    int yn = (i + L) % L2 < L ? i + L - L2 : i + L;

    int zp = i < L2 ? i - L2 + L3 : i - L2;
    int zn = i + L2 >= L3 ? i + L2 - L3 : i + L2;

    /* intermedia matrices; A, B with J */
    Rt3 A, B, C, D, R_try;
    double rdE; // reverse of energy change

    /* local energy = - Tr(C * R[i]); [U, J] = 0 */
    A = (R[xp].transpose() * Ux[xp] + R[yp].transpose() * Uy[yp]
         + R[zp].transpose() * Uz[zp]) * J;
    B = J * (Ux[i] * R[xn] + Uy[i] * R[yn] + Uz[i] * R[zn]);
    C = A + B.transpose();

    /* multi-hits */
    for(int j = 0; j < hits_R; j++)
    {
        random_R(R_try);
        D = C * (R_try - R[i]);
        rdE = D.trace();

        if( rdE > 0 || exp(beta * rdE) > random_01(rng) )
        {
            current_energy -= rdE;
            R[i] = R_try;
            flip_counter[0]++;
        }
    }
}


/** Update of U in x-bond,
 * relevant bond (i, xn)
 * **/
void gauge_sim::flip_Ux(int i)
{
    int xn = (i + 1) % L == 0 ? i + 1 - L : i + 1;

    /* intermedia value */
    Rt3 A, B, U_try;
    double rdE;

    A = R[xn] * R[i].transpose() * J;

    for(int j = 0; j < hits_U; j++)
    {
        U_try = gauge_bath[random_U(rng)];
        B = A * (U_try - Ux[i]);
        rdE = B.trace();

        if( rdE > 0 || exp(beta*rdE) > random_01(rng) )
        {
            current_energy -= rdE;
            Ux[i] = U_try;
            flip_counter[1]++;
        }

    }

}


/** Update gauge field in y-bond,
 * relevant bond (i, yn),
 **/
void gauge_sim::flip_Uy(int i)
{
    int yn = (i + L) % L2 < L ? i + L - L2 : i + L;

    Rt3 A, B, U_try;
    double rdE;

    A = R[yn] * R[i].transpose() * J;

    for(int j = 0; j < hits_U; j++)
    {
        U_try = gauge_bath[random_U(rng)];
        B = A * (U_try - Uy[i]);
        rdE = B.trace();

        if( rdE > 0 || exp(beta * rdE) > random_01(rng) )
        {
            current_energy -= rdE;
            Uy[i] = U_try;
            flip_counter[2]++;
        }
    }
}


/** Update gauge field in z-bond,
 * relevant bond (i, zn),
 **/
void gauge_sim::flip_Uz(int i)
{
    int zn = i + L2 >= L3 ? i + L2 - L3 : i + L2;

    Rt3 A, B, U_try;
    double rdE;

    A = R[zn] * R[i].transpose() * J;

    for(int j = 0; j < hits_U; j++)
    {
        U_try = gauge_bath[random_U(rng)];
        B = A * (U_try - Uz[i]);
        rdE = B.trace();

        if( rdE > 0 || exp(beta*rdE) > random_01(rng) )
        {
            current_energy -= rdE;
            Uz[i] = U_try;
            flip_counter[3]++;
        }
    }

}


double gauge_sim::total_energy() {

    int xn, yn, zn;
    Rt3 A, E;

    E = Eigen::MatrixXd::Zero(3,3);

    for (int i = 0; i < L3; ++i)
    {
        /* Energy of three local bonds at i*/
        xn = (i + 1) % L == 0 ? i + 1 - L : i + 1;
        yn = (i + L) % L2 < L ? i + L - L2 : i + L;
        zn = i + L2 >= L3 ? i + L2 - L3 : i + L2;

        A = R[i].transpose() * J * (Ux[i] * R[xn] + Uy[i] * R[yn] + Uz[i] * R[zn]);
        E += A;
    }

    return -E.trace();
}


void gauge_sim::random_R(Rt3& Rt) {

    /* Generate x0, x1,x2 in [0,1) */

    double x0, x1, x2;


    x0 = random_01(rng);
    x1 = random_01(rng);
    x2 = random_01(rng);

    double g1, s1, c1, r1, g2, s2, c2, r2;

    g1 = pi2*x1; s1 = sin(g1); c1 = cos(g1);
    g2 = pi2*x2; s2 = sin(g2); c2 = cos(g2);

    r1 = sqrt(1-x0);
    r2 = sqrt(x0);

    /* coordinate of the 4d vector*/
    double q0, q1, q2, q3;

    q0 = c2*r2;
    q1 = s1*r1;
    q2 = c1*r1;
    q3 = s2*r2;

    /* Define the rotational matrix */


    double q11, q22, q33, q01, q02, q03, q12, q13, q23;

    q11 = q1*q1; q22 = q2*q2; q33 = q3*q3;
    q01 = q0*q1; q02 = q0*q2; q03 = q0*q3;
    q12 = q1*q2; q13 = q1*q3; q23 = q2*q3;

    Rt << 1 - 2*(q22+q33), 2*(q12 - q03), 2*(q13 + q02),
        2*(q12 + q03), 1-2*(q33 + q11), 2*(q23-q01),
        2*(q13-q02), 2*(q23+q01), 1-2*(q11+q22);

    if(!O3)
    { return; }
    else
    { Rt *= 2 * random_int_01(rng) - 1; }
}

// Collects the measurements at each MC step,
//after thermailization has been done
void gauge_sim::measure() {
    if (sweeps < thermalization_sweeps) return;

    double E = current_energy/Eg;
    //double E2 = E*E;
    double nem2 = nematicity();
    double nem = sqrt(nem2);

    // Accumulate the data
    measurements["Energy"] << E;
    //measurements["Energy^2"] << E2;
    //measurements["Energy^3"] << E2*E;

    measurements["Nematicity"] << nem;
    //measurements["Nematicity^2"] << nem2;
    //measurements["Nematicity^4"] << nem2 * nem2;
}

// Returns a number between 0.0 and 1.0 with the completion percentage
double gauge_sim::fraction_completed() const {
    double f=0;
    if (total_sweeps>0 && sweeps >= thermalization_sweeps) {
        f=(sweeps-thermalization_sweeps)/double(total_sweeps);
    }
    return f;
}

// Saves the state to the hdf5 file
/** save the check point **/
void gauge_sim::save(alps::hdf5::archive & ar) const {
    // Most of the save logic is already implemented in the base class
    alps::mcbase::save(ar);

    /* Save the flip ratios */
    double trying_R = 0.01 * sweeps * sweep_unit * hits_R * L3;
    double trying_U = 0.01 * sweeps * sweep_unit * hits_U * L3;

    std::vector<double> flip_ratio(flip_counter.size());
    flip_ratio[0] = flip_counter[0]/trying_R;
    flip_ratio[1] = flip_counter[1]/trying_U;
    flip_ratio[2] = flip_counter[2]/trying_U;
    flip_ratio[3] = flip_counter[3]/trying_U;

    ar["checkpoint/flip_counter"] << flip_counter;
    ar["checkpoint/flip_ratio"] << flip_ratio;

    /* Save the configuration */

    boost::multi_array<double, 2> R_data;
    boost::multi_array<double, 2> Ux_data;
    boost::multi_array<double, 2> Uy_data;
    boost::multi_array<double, 2> Uz_data;

    R_data.resize(boost::extents[L3][9]);
    Ux_data.resize(boost::extents[L3][9]);
    Uy_data.resize(boost::extents[L3][9]);
    Uz_data.resize(boost::extents[L3][9]);

    for(int i = 0; i < L3; i++)
        for(int j = 0; j < 9; j++)
        {
            R_data[i][j] = R[i](j);
            Ux_data[i][j] = Ux[i](j);
            Uy_data[i][j] = Uy[i](j);
            Uz_data[i][j] = Uz[i](j);
        }


    ar["checkpoint/configuration/R"] << R_data;
    ar["checkpoint/configuration/Ux"] << Ux_data;
    ar["checkpoint/configuration/Uy"] << Uy_data;
    ar["checkpoint/configuration/Uz"] << Uz_data;

}

// Loads the state from the hdf5 file
void gauge_sim::load(alps::hdf5::archive & ar) {
    // Most of the load logic is already implemented in the base class
    alps::mcbase::load(ar);

    /*Load the configuration from checkpoint*/
    boost::multi_array<double, 2> R_data;
    boost::multi_array<double, 2> Ux_data;
    boost::multi_array<double, 2> Uy_data;
    boost::multi_array<double, 2> Uz_data;

    R_data.resize(boost::extents[L3][9]);
    Ux_data.resize(boost::extents[L3][9]);
    Uy_data.resize(boost::extents[L3][9]);
    Uz_data.resize(boost::extents[L3][9]);

    ar["checkpoint/configuration/R"] >> R_data;
    ar["checkpoint/configuration/Ux"] >> Ux_data;
    ar["checkpoint/configuration/Uy"] >> Uy_data;
    ar["checkpoint/configuration/Uz"] >> Uz_data;


    for(int i = 0; i < L3; i++)
        for(int j = 0; j < 9; j++)
        {
            R[i](j) = R_data[i][j];
            Ux[i](j) = Ux_data[i][j];
            Uy[i](j) = Uy_data[i][j];
            Uz[i](j) = Uz_data[i][j];
        }

}

void gauge_sim::reset_sweeps(bool skip_therm) {
    if (skip_therm)
        sweeps = thermalization_sweeps;
    else
        sweeps = 0;
}

void gauge_sim::temperature(double new_temp) {
    parameters["temperature"] = new_temp;
    beta = 1. / new_temp;
}

bool gauge_sim::is_thermalized() const {
    return sweeps > thermalization_sweeps;
}

size_t gauge_sim::configuration_size() const {
    return confpol->size();
}

std::vector<double> gauge_sim::configuration() const {
    return confpol->configuration(R);
}
