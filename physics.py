import numpy as np
import copy        
class ElectromagneticEquations:                            
    def __init__(self, c = 299792458):
        self.c = c
        self.EPSILON_0 = 8.8541878128e-12 
        self.epsilon = np.finfo(np.float64).eps # ~ 2.22e-16
        self.MU_0 = 4* np.pi * 10**-7    #= 4 * np.pi * 10**-7    
        self.ELEMENTARY_CHARGE = 1.602176634*1e-19
        self.e_mass = 9.10*1e-31  
        self.p_mass = 1.67*1e-27           
        self.k =8.9875517873681764e9
    #########################################
    # FORCE
    #########################################
    def Lorentz(self, particle, position, v):
        v = particle.velocity.copy()
        gamma = self.Gamma(v)  # Calculate the relativistic gamma factor
        magnitude =(particle.charge / (gamma * particle.mass)) 
        acceleration  = magnitude* (particle.total_E_field + np.cross(v, particle.total_B_field))
        return acceleration
        
    #########################################          
    def Landau_Lifshitz(self,particle, pos, velocity):
        v=velocity.copy()
        E = particle.total_E_field.copy()
        B = particle.total_B_field.copy()
        q = particle.charge
        m = particle.mass
        g = self.Gamma(v)         
        if np.allclose(v, 0):
            # Handle the case when v is zero
            v_norm = np.zeros_like(v)
            return (q / m) * E
        else:
            v_norm = v / np.linalg.norm(v)
            Landau_Lifshitz = (q / (g * m)) * (E + np.cross(v, B) / g - (q / (g * m * self.c**2)) * np.dot(E, v) * v)
            rad_const = (q**3 / (6 * np.pi * m**2 * self.c**3))            
            Landau_rad = (-(g**2 / self.c**2) * np.dot(E, E) * v + (q / m) * np.cross(E, B) + (q / (m * self.c**2)) * np.dot(E, v) * B   ) 
        return (Landau_Lifshitz + rad_const * Landau_rad)     
        
    #########################################
    def landau_lifshitz_force(self,particle, pos, velocity ):
        """
        Calculates the combined Lorentz and simplified radiation reaction forces on a particle.
        """
        v=velocity.copy()
        q = particle.charge
        m = particle.mass
        gamma = particle.gamma  # Assuming gamma is pre-calculated
        E = particle.total_E_field.copy() 
        B = particle.total_B_field.copy()         
        lorentz_force = q * (E + np.cross(v, B))
        # Simplified radiation reaction component
        # This simplification assumes a proportional damping force opposing the velocity
        rad_reaction_factor = 2 * q**3 / (3 * m * self.c**3)
        rad_reaction_force = -rad_reaction_factor * gamma**2 * np.dot(v, E) * v  # Simplified model
        # Combine forces
        total_force = lorentz_force + rad_reaction_force        
        return total_force / (m * gamma)  
        
    #########################################       
    # VELOCITY
    #########################################    
    def rk4_step(self, p):
        dt = p.dt
        position = p.position.copy()
        velocity = p.velocity.copy()
        force_func = p.force_method
        k1_v = velocity
        k1_a = force_func( position, velocity )
        k2_v = velocity + k1_a * dt / 2
        k2_a = force_func( position + k1_v * dt / 2, k2_v )
        k3_v = velocity + k2_a * dt / 2
        k3_a = force_func( position + k2_v * dt / 2, k3_v )
        k4_v = velocity + k3_a * dt
        k4_a = force_func( position + k3_v * dt, k4_v )
        new_velocity = velocity + (k1_a + 2*k2_a + 2*k3_a + k4_a) * dt / 6
        new_position = position + (k1_v + 2*k2_v + 2*k3_v + k4_v) * dt / 6    
        return new_position, new_velocity
        
    #########################################
    def rk6_step(self,p):
        dt = p.dt
        position = p.position.copy()
        velocity = p.velocity.copy()
        force_func = p.force_method  
        k1_v = velocity
        k1_a = force_func(position, velocity)        
        k2_v = velocity + k1_a * dt / 6
        k2_a = force_func(position + k1_v * dt / 6, k2_v)        
        k3_v = velocity + (3*k1_a + 9*k2_a) * dt / 40
        k3_a = force_func(position + (3*k1_v + 9*k2_v) * dt / 40, k3_v)        
        k4_v = velocity + (3*k1_a - 9*k2_a + 12*k3_a) * dt / 10
        k4_a = force_func(position + (3*k1_v - 9*k2_v + 12*k3_v) * dt / 10, k4_v)        
        k5_v = velocity + (-11*k1_a + 135*k2_a + 140*k3_a + 70*k4_a) * dt / 54
        k5_a = force_func(position + (-11*k1_v + 135*k2_v + 140*k3_v + 70*k4_v) * dt / 54, k5_v)        
        k6_v = velocity + (1631*k1_a + 175*k2_a + 575*k3_a + 44275*k4_a + 253*k5_a) * dt / 55296
        k6_a = force_func(position + (1631*k1_v + 175*k2_v + 575*k3_v + 44275*k4_v + 253*k5_v) * dt / 55296, k6_v)        
        # Final position and velocity
        new_velocity = velocity + (37*k1_a + 375*k2_a + 1500*k3_a + 2500*k4_a + 625*k5_a + 512*k6_a) * dt / 4480
        new_position = position + (37*k1_v + 375*k2_v + 1500*k3_v + 2500*k4_v + 625*k5_v + 512*k6_v) * dt / 4480        
        return new_position, new_velocity  

    #########################################       
    def analytic_rel(self, p):
        # Analytic solution considering relativistic velocities
        B = p.total_B_field.copy()
        dt = p.dt
        m = p.mass  
        q = -p.charge
        vel = p.velocity.copy()
        pos = p.position.copy()
        c = self.c 
        def rotate_vector(vector, axis, angle):
            K = np.array([[0, -axis[2], axis[1]], 
                          [axis[2], 0, -axis[0]], 
                          [-axis[1], axis[0], 0]])
            rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            return rotation_matrix @ vector    
        gamma = self.Gamma(vel) 
        omega = q * np.linalg.norm(B) / (gamma * m)
        B_unit = B / np.linalg.norm(B)    
        vel_parallel = np.dot(vel, B_unit) * B_unit
        vel_perp = vel - vel_parallel
        vel_perp_rotated = rotate_vector(vel_perp, B_unit, omega * dt)    
        new_vel = vel_perp_rotated + vel_parallel 
        displacement_perp = vel_perp_rotated * np.sin(omega * dt) / omega - vel_perp * (1 - np.cos(omega * dt)) / omega
        new_pos = pos + displacement_perp + vel_parallel * dt    
        return new_pos, new_vel
        
    #########################################              
    def analytic(self, p):
        # Analytic solution considering classic velocities        
        B = p.total_B_field.copy()
        dt = p.dt
        m = p.mass  
        q = -p.charge
        vel = p.velocity.copy()
        pos = p.position.copy()
        def rotate_vector(vector, axis, angle):             
            K = np.array([[0, -axis[2], axis[1]], 
                          [axis[2], 0, -axis[0]], 
                          [-axis[1], axis[0], 0]])
            rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            return rotation_matrix @ vector    
        B_norm = np.linalg.norm(B)
        omega = q * B_norm / m
        B_unit = B / B_norm     
        vel_parallel = np.dot(vel, B_unit) * B_unit
        vel_perp = vel - vel_parallel
        vel_perp_rotated = rotate_vector(vel_perp, B_unit, omega * dt)    
        new_vel = vel_perp_rotated + vel_parallel 
        displacement_perp = vel_perp_rotated * np.sin(omega * dt) / omega - vel_perp * (1 - np.cos(omega * dt)) / omega
        new_pos = pos + displacement_perp + vel_parallel * dt    
        return new_pos, new_vel
        
    #########################################                  
    def boris_push(self, p):
        E = p.total_E_field.copy()
        B = p.total_B_field.copy()
        dt = p.dt
        m = p.mass
        q = p.charge
        position = p.position.copy()
        velocity = p.velocity.copy()    
        gamma = self.Gamma(velocity)    
        # Half-step velocity update due to electric field
        v_minus = velocity + (q / m) * E * (dt / 2.0) / gamma    
        # Boris rotation in the magnetic field
        t_vector = (q / m) * B * (dt / 2.0) / gamma
        s = 2 * t_vector / (1 + np.linalg.norm(t_vector)**2)
        v_prime = v_minus + np.cross(v_minus, t_vector)
        v_plus = v_minus + np.cross(v_prime, s)    
        # Check if the updated velocity exceeds the speed of light
        if np.linalg.norm(v_plus) > self.c:
            position += v_plus * dt  
            return position, v_plus  
        # Second half-step velocity update due to the electric field
        gamma = self.Gamma(v_plus)
        velocity = v_plus + (q / m) * E * (dt / 2.0) / gamma    
        position += velocity * dt    
        return position, velocity
        
    #########################################        
    def vay_push(self, p):
        E = p.total_E_field.copy()   
        B = p.total_B_field.copy()
        m = p.mass   
        q = p.charge
        x = p.position.copy()
        v = p.velocity.copy()
        dt = p.dt
        half_dt_q_over_m = q * dt / (2.0 * m)        
        gamma_v = self.Gamma(v)
        v_minus = v + half_dt_q_over_m * E / gamma_v        
        if np.linalg.norm(v_minus) > self.c:
            x += v_minus * dt  
            return x, v_minus  # Return position and v_plus
        gamma_v_minus = self.Gamma(v_minus)
        tt = half_dt_q_over_m * B / gamma_v_minus
        s = 2.0 * tt / (1.0 + np.sum(tt**2))
        v_prime = v_minus + np.cross(v_minus, tt)
        v_plus = v_minus + np.cross(v_prime, s)        
        if np.linalg.norm(v_plus) > self.c:
            x += v_plus * dt  
            return x, v_plus  # Return position and v_plus        
        gamma_v_plus = self.Gamma(v_plus)
        v_new = v_plus + half_dt_q_over_m * E / gamma_v_plus        
        x_new = x + v_new * dt        
        return x_new, v_new
        
    #########################################
    def relativ_intgrtr(self, p, integrator_type='symplectic'):
        E = p.total_E_field.copy()   
        B = p.total_B_field.copy()
        m = p.mass   
        q = p.charge
        x = p.position.copy()
        v = p.velocity.copy()
        dt = p.dt
        c = self.c             
        def magnetic_force(v, B):
            return q * np.cross(v, B)    
        def electric_force(E):
            return q * E    
        def update_velocity(v, a, dt):
            return v + a * dt    
        def update_position(r, v, dt):
            return r + v * dt    
        if integrator_type == 'hamiltonian':
            # Hamiltonian integrator
            v_half = v + 0.5 * dt * (electric_force(E) + magnetic_force(v, B)) / (m * self.Gamma(v))
            x = update_position(x, v_half, dt)
            v = update_velocity(v_half, (electric_force(E) + magnetic_force(v_half, B)) / (m * self.Gamma(v_half)), 0.5 * dt)
        elif integrator_type == 'symplectic':
            # Symplectic integrator
            v = update_velocity(v, (electric_force(E) + magnetic_force(v, B)) / (m * self.Gamma(v)), 0.5 * dt)
            x = update_position(x, v, dt)
            if np.linalg.norm(v) > self.c:
                x += v * dt  
                return x, v  
            v = update_velocity(v, (electric_force(E) + magnetic_force(v, B)) / (m * self.Gamma(v)), 0.5 * dt)
        else:
            raise ValueError("Invalid integrator type. Choose 'hamiltonian' or 'symplectic'.")    
        return x, v
        
    #########################################        
    def Hamiltonian(self, p):
        c = self.c#3e8  # Speed of light in meters per second
        E = p.total_E_field.copy()
        B = p.total_B_field.copy()
        m = p.mass
        q = p.charge
        r = p.position.copy()
        v = p.velocity.copy()
        dt = p.dt    
        gamma = self.Gamma(v) #1 / np.sqrt(1 - (np.linalg.norm(v) / c)**2)    
        # Calculate the relativistic momentum
        momentum = gamma * m * v    
        # Calculate the Lorentz force
        F = q * (E + np.cross(v, B))    
        # Update the momentum (half-step)
        p_half = momentum + (dt / 2) * F    
        # Update the velocity (half-step)
        gamma_half = np.sqrt(1 + (np.linalg.norm(p_half) / (m * c))**2)
        v_half = p_half / (gamma_half * m)    
        # Update the position
        r_new = r + dt * v_half    
        # Update the momentum (full-step)
        F_new = q * (E + np.cross(v_half, B))
        p_new = p_half + (dt / 2) * F_new    
        gamma_new = np.sqrt(1 + (np.linalg.norm(p_new) / (m * c))**2)
        v_new = p_new / (gamma_new * m)    
        return r_new, v_new

    #########################################        
    def vay_algorithm(self, p):
        E = p.total_E_field  # Assume these fields are numpy arrays
        B = p.total_B_field
        m = p.mass
        q = p.charge
        dt = p.dt
        x = p.position.copy()
        v = p.velocity.copy()    
        gamma = self.Gamma(v)    
        # Calculate the intermediate velocity v_prime
        v_prime = v + (q * dt / (2.0 * m * gamma)) * (E + np.cross(v, B))        
        if np.linalg.norm(v_prime) > self.c:
                x += v_prime * dt  # Update position with v_plus before returning            
                return x, v_prime  # Return position and v_plus
        gamma_prime = self.Gamma(v_prime)        
        # Update the velocity using the magnetic field rotation step
        tt = (q * dt / (2.0 * m * gamma_prime)) * B
        s = 2.0 * tt / (1.0 + np.linalg.norm(tt)**2)
        v_new = v_prime + np.cross(v_prime + np.cross(v_prime, tt), s)    
        x_new = x + dt * (v + v_new) / 2.0        
        return x_new, v_new  
        
    #########################################
    # ENERGY LOSS
    #########################################
    def synchrotron_radiation(self, particle, magnetic_field, v, dt):
        velocity = v.copy()  # Using copy to avoid modifying the input velocity outside the function
        # Normalize the magnetic field -> unit vector
        magnetic_field_mag = np.linalg.norm(magnetic_field)
        if magnetic_field_mag == 0 or np.linalg.norm(velocity) == 0:
            return velocity, 0, particle.mass * self.c ** 2    
        magnetic_field_unit = magnetic_field / magnetic_field_mag    
        # Velocity parallel and perpendicular to the magnetic field components
        v_parallel = np.dot(velocity, magnetic_field_unit) * magnetic_field_unit
        v_perpendicular = velocity - v_parallel
        v_perp_mag = np.linalg.norm(v_perpendicular)    
        # No perpendicular component to lose energy from
        if v_perp_mag == 0:
            return velocity, 0, particle.total_energy    
        # Calculate the radius of the particle's path
        r = particle.gamma * particle.mass * v_perp_mag / (np.abs(particle.charge) * magnetic_field_mag)
        omega_c = (3/2) * particle.gamma**3 * (self.c / r)
        # Convert the critical frequency in Hz
        f_c = omega_c / (2 * np.pi)    
        # Power radiated due to synchrotron radiation
        a_perp = v_perp_mag ** 2 / r if r > 0 else 0
        P = (2.0 / 3) * (particle.charge ** 2 * a_perp ** 2) / (4 * np.pi * self.EPSILON_0 * self.c ** 3)  
        # Adjust velocity perpendicular component and energy based on energy loss
        new_energy = particle.total_energy - P * dt
        rest_mass_energy = particle.mass * self.c ** 2    
        if new_energy < rest_mass_energy:
            new_energy = rest_mass_energy
            adjusted_velocity = np.zeros_like(velocity)  # Particle effectively stops moving
        else:
            new_gamma = new_energy / rest_mass_energy
            new_v_mag = self.c * np.sqrt(1 - 1 / (new_gamma ** 2))    
            if v_perp_mag != 0:
                adjusted_v_perpendicular = (new_v_mag / v_perp_mag) * v_perpendicular
            else:
                adjusted_v_perpendicular = np.zeros_like(velocity)    
            adjusted_velocity = v_parallel + adjusted_v_perpendicular    
            # Ensuring the new velocity magnitude does not exceed speed of light
            if np.linalg.norm(adjusted_velocity) >= self.c:
                adjusted_velocity = adjusted_velocity / np.linalg.norm(adjusted_velocity) * (self.c - self.epsilon)    
        return adjusted_velocity, P * dt, new_energy
        
    ####################################     
    def radiation_reaction1(self, particles, dt):
        # Assuming particles is a structured array or has attributes like arrays
        q = particles.charge  # Array of charges
        m = particles.mass    # Array of masses
        acceleration = particles.acc  # Array of accelerations
        velocities = particles.velocity  # Array of velocities    
        gamma = self.Gamma(velocities)
        a_magnitude = np.linalg.norm(acceleration, axis=1)    
        P = (q**2 * a_magnitude**2 * gamma**6) / (6 * np.pi * self.EPSILON_0 * self.c**3)
        delta_E = P * dt
        KE_initial = (gamma - 1) * m * self.c**2
        KE_final = KE_initial - delta_E    
        gamma_new = np.where(KE_final > 0, (KE_final / (m * self.c**2)) + 1, 1)
        v_magnitude_new = self.c * np.sqrt(1 - 1 / (gamma_new**2))    
        # Handling direction when velocity might be zero
        v_norm = np.linalg.norm(velocities, axis=1)
        v_direction = np.where(v_norm[:, np.newaxis] > 0, velocities / v_norm[:, np.newaxis], 0)
        v_new = v_direction * v_magnitude_new[:, np.newaxis]    
        # Ensure no new velocity is calculated when kinetic energy is completely lost
        v_new = np.where(KE_final[:, np.newaxis] > 0, v_new, 0)
        return v_new, delta_E, KE_final
    
    def radiation_reaction(self, particle, B, v, dt ):
        # The radiation_reaction function is generally more accurate 
        q= particle.charge
        m=particle.mass
        acceleration = particle.acc.copy()         
        gamma = self.Gamma(v) 
        a_magnitude = np.linalg.norm(acceleration)         
        P = (q**2 * a_magnitude**2 * gamma**6) / (6 * np.pi * self.EPSILON_0  * self.c**3)
        # Energy lost in time dt
        delta_E = P * dt 
        # Update the particle's kinetic energy
        KE_initial = (gamma - 1) * m * self.c**2         
        KE_final = KE_initial - delta_E
        if KE_final > 0:
            # Solve for new gamma after energy loss
            gamma_new = (KE_final / (m * self.c**2)) + 1
            gamma_new = max(gamma_new, 1)             
            energy_loss_threshold = 1e-10  # Adjust this value based on your requirements
            v_magnitude_new = self.c * np.sqrt(1 - 1 / (gamma_new**2))
            v_direction = v / np.linalg.norm(v)
            v_new = v_direction * v_magnitude_new                
        else:
            v_new = np.array([0.0, 0.0, 0.0])
            KE_final= m * self.c ** 2
        return v_new, delta_E, KE_final                 
       
    #########################################     
    # particles interaction vectorized
    #########################################   
    def calculate_retardation_vectorized(self,observer_position, particle_positions):       
        r_vectors = particle_positions - observer_position                
        distances = np.linalg.norm(r_vectors, axis=1)                
        retarded_times = distances / self.c          
        return r_vectors, retarded_times    
        
    #########################################   
    def retarded_state_vectorized(self, particles, retarded_times):
        # Ensure inputs are arrays for vectorized operations
        if not isinstance(retarded_times, np.ndarray):
            retarded_times = np.array(retarded_times)    
        num_particles = len(particles)
        pos_final = np.zeros((num_particles, 3)) 
        vel_final = np.zeros((num_particles, 3))
        acc_final = np.zeros((num_particles, 3))    
        # Loop to vectorize (minimal loop over particles still needed if data structure varies per particle)
        for i, particle in enumerate(particles):
            indices = np.searchsorted(particle.cumulative_times, retarded_times[i], side='right') - 1
            indices = np.clip(indices, 0, None)  # Ensure indices are not negative    
            pos_final[i] = particle.pos_traj[indices] if indices < len(particle.pos_traj) else particle.pos_traj[-1]
            vel_final[i] = particle.vel_traj[indices] if indices < len(particle.vel_traj) else particle.vel_traj[-1]
            acc_final[i] = particle.acc_traj[indices] if indices < len(particle.acc_traj) else particle.acc_traj[-1]    
        return pos_final, vel_final, acc_final 
        
    #########################################         
    def calculate_retarded_distances_vectorized(self, observer_position, retarded_positions):
        r_retarded = observer_position - retarded_positions
        r_retarded_mag = np.linalg.norm(r_retarded, axis=1)        
        # Create a mask where the distances are exactly zero
        zero_distance_mask = (r_retarded_mag == 0)        
        r_retarded_unit = np.zeros_like(r_retarded)        
        # Calculate unit vectors only where there are non-zero distances
        non_zero_distances = ~zero_distance_mask
        r_retarded_unit[non_zero_distances] = r_retarded[non_zero_distances] / r_retarded_mag[non_zero_distances, np.newaxis]        
        # explicitly set zero vectors for zero distances:
        r_retarded_unit[zero_distance_mask] = 0    
        r_retarded[zero_distance_mask] = 0
        r_retarded_mag[zero_distance_mask] = 0    
        return r_retarded, r_retarded_mag, r_retarded_unit
        
    ######################################### 
    def calculate_magnetic_field_vectorized(self, charges, r_retarded_mag, r_retarded_unit, retarded_velocities):
        B_dir = np.cross(r_retarded_unit, retarded_velocities)
        B_dir_mag = np.linalg.norm(B_dir, axis=1, keepdims=True)    
        B_dir_unit = np.zeros_like(B_dir)    
        # Only perform division where B_dir_mag is greater than zero
        safe_indices = B_dir_mag[:, 0] > 0  
        B_dir_unit[safe_indices] = B_dir[safe_indices] / B_dir_mag[safe_indices]    
        # Magnetic field magnitude using the Biot-Savart Law
        norm_product = np.linalg.norm(retarded_velocities, axis=1) * np.linalg.norm(r_retarded_unit, axis=1) + np.finfo(float).eps
        cos_theta = np.einsum('ij,ij->i', r_retarded_unit, retarded_velocities) / norm_product
        cos_theta = np.clip(cos_theta, -1, 1)  # Ensuring cos_theta is within the valid range for sqrt
        sin_theta = np.sqrt(np.clip(1 - cos_theta**2, 0, 1))    
        B_mag = (self.MU_0 / (4 * np.pi)) * (charges * np.linalg.norm(retarded_velocities, axis=1) * sin_theta) / (r_retarded_mag ** 2 + np.finfo(float).eps)
        return B_mag[:, None] * B_dir_unit  
        
    #########################################             
    def calculate_electric_field_vectorized(self, charges, r_retarded, r_retarded_mag,
                                            r_retarded_unit, retarded_velocities, retarded_accs):
        epsilon = np.finfo(float).eps 
        beta = retarded_velocities / self.c
        gamma_ret = self.Gamma_vec(beta)
        one_minus_beta_dot_r = 1 - np.einsum('ij,ij->i', beta, r_retarded_unit)        
        # Identify safe indices where no divisions by zero occur
        safe_indices = (r_retarded_mag > epsilon) & (np.abs(one_minus_beta_dot_r) > epsilon)        
        # Filter all arrays to include only safe indices
        filtered_charges = charges[safe_indices]
        filtered_r_retarded_mag = r_retarded_mag[safe_indices]
        filtered_r_retarded_unit = r_retarded_unit[safe_indices]
        filtered_beta = beta[safe_indices]
        filtered_gamma_ret = gamma_ret[safe_indices]
        filtered_one_minus_beta_dot_r = one_minus_beta_dot_r[safe_indices]
        filtered_retarded_accs = retarded_accs[safe_indices]        
        common_factor = (filtered_charges / (4 * np.pi * self.EPSILON_0 * filtered_r_retarded_mag**2))[:, None]
        # Calculate velocity and acceleration terms only for safe indices
        velocity_term = (filtered_r_retarded_unit - filtered_beta) / (filtered_gamma_ret[:, None]**2 * filtered_one_minus_beta_dot_r[:, None]**3)
        acc_term = np.cross(filtered_r_retarded_unit, np.cross(filtered_r_retarded_unit - filtered_beta, filtered_retarded_accs / self.c)) / (self.c * filtered_one_minus_beta_dot_r[:, None]**3)
        electric_field = common_factor * (velocity_term + acc_term)        
        # Initialize full electric field array
        full_electric_field = np.zeros((charges.shape[0], 3))
        full_electric_field[safe_indices] = electric_field        
        return full_electric_field          

    #########################################    
    # Interaction non-vectorized
    #########################################         
    def calculate_retardation(self,self_p, particle):
        r = self_p.position - particle.position 
        retarded_time = np.linalg.norm(r) / self.c
        return r, retarded_time    
        
    def retarded_state(self, particles, retarded_time):
        def find_closest_index(cumulative_times, target_time):
            idx = np.searchsorted(cumulative_times, target_time, side='right') - 1
            return max(0, idx)  # Ensure the index is not negative    
        def retrieve_state(particle, index):
            # Safeguard against index out of bounds
            if index < len(particle.pos_traj):
                return particle.pos_traj[index], particle.vel_traj[index], particle.acc_traj[index]
            return particle.pos_traj[-1], particle.vel_traj[-1], particle.acc_traj[-1]    
        # Convert to list if a single particle is passed
        if not isinstance(particles, list):
            particles = [particles]        
        # Assuming cumulative_times are maintained as part of particle updates
        retarded_states = []
        for particle in particles:
            index = find_closest_index(particle.cumulative_times, retarded_time)
            state = retrieve_state(particle, index)
            retarded_states.append(state)    
        # Return a list of states or a single state depending on the input
        return retarded_states if len(retarded_states) > 1 else retarded_states[0]
    
    def calculate_retarded_distance(self, particle, retarded_position):
        r_retarded = particle.position - retarded_position
        r_retarded_mag = np.linalg.norm(r_retarded)
        if r_retarded_mag > 0:
            r_retarded_unit = r_retarded / r_retarded_mag
        else:
            r_retarded_unit = np.zeros_like(r_retarded)
        return r_retarded, r_retarded_mag, r_retarded_unit    
        
    #########################################           
    def calculate_electric_field(self, particle, r_retarded, r_retarded_mag, r_retarded_unit, retarded_velocity, retarded_acc):
        # Calculate beta and gamma for the retarded velocity
        beta = retarded_velocity/ self.c
        #beta = limit_speed(beta)
        gamma_ret = self.Gamma(beta)
        one_minus_beta_dot_r = 1 - np.dot(beta, r_retarded_unit)        
        # Precompute common factors used in the field calculation
        common_factor = particle.charge / (4 * np.pi * self.EPSILON_0 * (r_retarded_mag ** 2))
        velocity_term = (r_retarded_unit - beta) / (gamma_ret **2 * one_minus_beta_dot_r ** 3)
        acc_term = np.cross(r_retarded_unit, np.cross((r_retarded_unit - beta),
                                                      retarded_acc / self.c)) / (self.c * one_minus_beta_dot_r ** 3)        
        # Combine the velocity and acceleration terms
        return common_factor * (velocity_term + acc_term)
        
    #########################################   
    def calculate_magnetic_field(self, particle, r_retarded_mag, r_retarded_unit, retarded_velocity):
        # Calculate the direction of the magnetic field using the cross product
        B_dir = np.cross(r_retarded_unit, retarded_velocity)
        if np.linalg.norm(B_dir) > 0:
            B_dir_unit = B_dir / np.linalg.norm(B_dir)
        else:
            B_dir_unit = np.zeros(3)    
        # Compute the magnetic field magnitude using the Biot-Savart Law
        norm_product = np.linalg.norm(retarded_velocity) * np.linalg.norm(r_retarded_unit)
        if norm_product > 0:
            theta = np.arccos(np.clip(np.dot(r_retarded_unit, retarded_velocity) / norm_product, -1, 1))
        else:
            theta = 0        
        B_mag = (self.MU_0 / (4 * np.pi)) * (particle.charge * np.linalg.norm(retarded_velocity) * np.sin(theta)) / (r_retarded_mag ** 2)        
        # Return the magnetic field vector
        return B_mag * B_dir_unit           
        
    #########################################         
    # Lorentz transformation
    #########################################           
    def lorentz_transform_fields(self, E, B, v):
        E = np.array(E)
        B = np.array(B)
        v = np.array(v)    
        # Validate shapes (for single vectors, this check is more straightforward)
        if v.ndim != 1 or len(v) != 3:
            raise ValueError("Velocity must be 1D with three elements.")
        if E.shape[-1] != 3 or B.shape[-1] != 3:
            raise ValueError("Electric and magnetic fields must have the last dimension size 3.")    
        gamma = self.Gamma(v) 
        v_norm = np.linalg.norm(v)                     
        # Ensure v is broadcastable over the E and B arrays
        v_expanded = v.reshape((1,)*E.ndim + (3,))    
        # Lorentz transformations for electric and magnetic fields
        E_prime = gamma * (E + np.cross(v_expanded, B, axis=-1)) - \
                  gamma**2 / (gamma + 1) * np.sum(v_expanded * E, axis=-1, keepdims=True) * v_expanded / self.c**2
        B_prime = gamma * (B - np.cross(v_expanded, E, axis=-1) / self.c**2) - \
                  gamma**2 / (gamma + 1) * np.sum(v_expanded * B, axis=-1, keepdims=True) * v_expanded / self.c**2            
        return E_prime, B_prime   

    #########################################
    # Gamma speed limit
    #########################################
    def Gamma_vec(self, velocities):
        velocities = np.array(velocities)
        if velocities.ndim == 1:
            velocities = velocities[np.newaxis, :]  # Ensure velocities is always 2D
        v_squared = np.sum(velocities**2, axis=1) / self.c**2
        gammas = 1 / np.sqrt(1 - v_squared)
        return gammas  # Always return an ndarray
        
    #########################################   
    def Gamma(self, velocities):         
        velocities = np.array(velocities)
        if velocities.ndim == 1:
            velocities = velocities[np.newaxis, :]                                                     
        v_squared = np.sum(velocities**2, axis=1) / self.c**2    
        gammas = 1 / np.sqrt(1 - v_squared)
        return gammas if gammas.size > 1 else gammas.item()  

    def limit_speed(self,v, limit= 0.9999999):             
        v_copy = v.copy()
        speed = np.linalg.norm(v_copy)
        if speed > (self.c * limit): 
            return (v_copy / speed) * (limit * self.c)  
        return v_copy        


  
    

 

 

 