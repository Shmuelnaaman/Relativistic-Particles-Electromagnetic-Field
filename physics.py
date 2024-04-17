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
    #########################################
    #########################################
    def Lorentz(self, particle, position, v):
        """
        Calculate the acceleration due to the Lorentz force for a charged particle.
        Parameters:
        - charge: Charge of the particle.
        - mass: Mass of the particle.
        - total_E_field: Electric field vector (numpy array).
        - total_B_field: Magnetic field vector (numpy array).
        - v: Velocity vector of the particle (numpy array).
        Returns:
        - Acceleration vector due to the Lorentz force.
        """
        v = particle.velocity.copy()
        #adding the speed limit here cause error
        #v = self.limit_speed(particle.velocity)  # Ensure velocity doesn't exceed c or a set limit
        gamma = self.Gamma(v)  # Calculate the relativistic gamma factor
        # Calculate the acceleration from the Lorentz force         
        acceleration  = (particle.charge / (gamma * particle.mass)) * (particle.total_E_field + np.cross(v, particle.total_B_field))
        return acceleration
        
    #########################################       
    def analytic(self, p):
        """
        Calculates the position and velocity of a charged particle in a uniform magnetic field
        at a given time, using numpy arrays for position and velocity. 
        """        
        B = p.total_B_field.copy()
        t = p.dt
        m = p.mass  
        q = -p.charge
        vel = p.velocity.copy()
        pos = p.position.copy()
        def rotate_vector(vector, axis, angle):
            """Helper function to create a rotation matrix from an axis and angle (Rodrigues' rotation formula). """
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
        vel_perp_rotated = rotate_vector(vel_perp, B_unit, omega * t)    
        new_vel = vel_perp_rotated + vel_parallel 
        displacement_perp = vel_perp_rotated * np.sin(omega * t) / omega - vel_perp * (1 - np.cos(omega * t)) / omega
        new_pos = pos + displacement_perp + vel_parallel * t    
        return new_pos, new_vel
        
    #########################################


    def boris_push(self, p):
        E = p.total_E_field.copy()
        B = p.total_B_field.copy()
        t = p.dt
        m = p.mass
        q = p.charge
        position = p.position.copy()
        velocity = p.velocity.copy()
        q_over_m = q / m        
        # Calculate gamma before the update
        gamma = self.Gamma(velocity) #1 / np.sqrt(1 - np.linalg.norm(velocity)**2)        
        # Half-step velocity update due to the electric field
        v_minus = velocity + q_over_m * E * (t / 2.0) / gamma        
        # Boris rotation in the magnetic field
        t_vector = q_over_m * B * (t / 2.0) / gamma
        s = 2 * t_vector / (1 + np.linalg.norm(t_vector)**2)
        v_prime = v_minus + np.cross(v_minus, t_vector)
        v_plus = v_minus + np.cross(v_prime, s)        
        # Second half-step velocity update due to the electric field
        velocity = v_plus + q_over_m * E * (t / 2.0) / gamma        
        # Update gamma after the velocity update
        gamma = self.Gamma(velocity)# 1 / np.sqrt(1 - np.linalg.norm(velocity)**2)        
        # Update position using the average of the initial and final velocities
        v_avg = (velocity + v_minus) / 2.0
        position += v_avg * t / gamma        
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
        # Half-acceleration
           

        gamma_v = self.Gamma(v)
        v_minus = v + half_dt_q_over_m * E / gamma_v        
        # Rotation
        gamma_v_minus = self.Gamma(v_minus)
        t = half_dt_q_over_m * B / gamma_v_minus
        s = 2.0 * t / (1.0 + np.sum(t**2))
        v_prime = v_minus + np.cross(v_minus, t)
        v_plus = v_minus + np.cross(v_prime, s)        
        # Half-acceleration
        gamma_v_plus = self.Gamma(v_plus)
        v_new = v_plus + half_dt_q_over_m * E / gamma_v_plus        
        # Update position
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
        # Calculate the Lorentz factor using the correct value of c
        gamma = self.Gamma(v) #1 / np.sqrt(1 - (np.linalg.norm(v) / c)**2)    
        # Calculate the relativistic momentum
        momentum = gamma * m * v    
        # Calculate the Lorentz force
        F = q * (E + np.cross(v, B))    
        # Update the momentum (half-step)
        p_half = momentum + (dt / 2) * F    
        # Update the velocity (half-step)
        # Need to recompute gamma because momentum has changed
        gamma_half = np.sqrt(1 + (np.linalg.norm(p_half) / (m * c))**2)
        v_half = p_half / (gamma_half * m)    
        # Update the position
        r_new = r + dt * v_half    
        # Update the momentum (full-step)
        F_new = q * (E + np.cross(v_half, B))
        p_new = p_half + (dt / 2) * F_new    
        # Update the velocity (full-step)
        gamma_new = np.sqrt(1 + (np.linalg.norm(p_new) / (m * c))**2)
        v_new = p_new / (gamma_new * m)    
        return r_new, v_new


    #########################################
        
    def vay_algorithm(self, p):
        # use average velocity as the new velocity to maintain energy
        E = p.total_E_field.copy()   
        B = p.total_B_field.copy()
        #t = p.dt
        m = p.mass   
        q = p.charge
        x = p.position.copy()
        v = p.velocity.copy()
        dt = p.dt     
        # Calculate the Lorentz factor gamma
        gamma = self.Gamma(v)
        # Calculate the intermediate velocity v_prime
        v_prime = v + (q * dt / (2.0 * m * gamma)) * (E + np.cross(v, B))    
        # Calculate the Lorentz factor gamma_prime
        gamma_prime = self.Gamma(v_prime)# 1.0 / np.sqrt(1.0 - np.sum(v_prime**2) / self.c**2)    
        # Calculate the new velocity v_new
        t = (q * dt / (2.0 * m * gamma_prime)) * B
        s = 2.0 * t / (1.0 + np.sum(t**2))
        v_new = v_prime + np.cross(v_prime + np.cross(v_prime, t), s)    
        # Calculate the new position x_new
        x_new = x + (dt / 2.0) * (v + v_new)    
        return x_new, v_new    
    
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
 
 
    def Landau_Lifshitz(self,particle, pos, velocity):
        v=velocity.copy()
        E = particle.total_E_field.copy()
        B = particle.total_B_field.copy()
        q = particle.charge
        m = particle.mass
        #The speed limit was causing issue in the Lorentz so I removed it here too
        #v = self.limit_speed(v)
        g = self.Gamma(v)         
        if np.allclose(v, 0):
            # Handle the case when v is zero
            # You can choose an appropriate fallback value or behavior
            v_norm = np.zeros_like(v)
            return (q / m) * E
        else:
            v_norm = v / np.linalg.norm(v)
            #print(np.dot(E, v_norm) * v, np.linalg.norm(E), np.linalg.norm(v))
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
        # Lorentz force component        
        lorentz_force = q * (E + np.cross(v, B))
        # Simplified radiation reaction component
        # This simplification assumes a proportional damping force opposing the velocity
        rad_reaction_factor = 2 * q**3 / (3 * m * self.c**3)
        rad_reaction_force = -rad_reaction_factor * gamma**2 * np.dot(v, E) * v  # Simplified model
        # Combine forces
        total_force = lorentz_force + rad_reaction_force        
        return total_force / (m * gamma)  
        
    #########################################
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
    #########################################
    
    def synchrotron_radiation1(self, particle, magnetic_field, v, dt):  
        velocity=vv.copy()
        # Normalize the magnetic field _> unit vector
        magnetic_field_mag = np.linalg.norm(magnetic_field)
        if magnetic_field_mag == 0 or np.linalg.norm(velocity) == 0:
            return velocity, 0, particle.mass * self.c ** 2            
        magnetic_field_unit = magnetic_field / magnetic_field_mag        
        # velocity parallel and perpendicular to the magnetic field components
        v_parallel = np.dot(velocity, magnetic_field_unit) * magnetic_field_unit
        v_perpendicular = velocity - v_parallel
        v_perp_mag = np.linalg.norm(v_perpendicular)        
        # No perpendicular component to lose energy from 
        if v_perp_mag == 0:
            return velocity, 0, particle.total_energy            
        # radius of the particle's path 
        r = particle.gamma * particle.mass * v_perp_mag / (np.abs(particle.charge) * magnetic_field_mag)         
        omega_c = (3/2) * particle.gamma**3 * (self.c / r)
        # Convert the critical frequency frequency in Hz
        f_c = omega_c / (2 * np.pi)            
        # Power radiated due to synchrotron radiation
        a_perp = v_perp_mag ** 2 / r if r > 0 else 0
        P = (2.0 / 3) * (particle.charge ** 2 * a_perp ** 2) / (4 * np.pi * self.EPSILON_0 * self.c ** 3)        
        # Adjust velocity perpendicular component and energy based on energy loss
        new_energy = particle.total_energy - P * dt
        if new_energy > 0:  #particle.mass * self.c ** 2:
            new_gamma = new_energy / (particle.mass * self.c ** 2)
            new_v_mag = self.c * np.sqrt(1 - 1 / (new_gamma ** 2))             
            if v_perp_mag != 0:
                adjusted_v_perpendicular = (new_v_mag / v_perp_mag) * v_perpendicular
            else:
                adjusted_v_perpendicular = np.zeros_like(velocity)
            v_total_mag = np.linalg.norm(v_parallel + adjusted_v_perpendicular)            
            if v_total_mag > self.epsilon:   
                adjusted_velocity = (v_parallel + adjusted_v_perpendicular) 
            else:
                adjusted_velocity = np.zeros_like(v_parallel) 
        else:  
            # Particle stops if energy is depleted
            adjusted_velocity = np.zeros_like(velocity)  
            new_energy =  particle.mass * self.c ** 2            
        return  adjusted_velocity, P * dt, new_energy    
        
    #########################################            
    
    def radiation_reaction1(self, particle, B, v, dt ):
        # The radiation_reaction function is generally more accurate 
        q= particle.charge
        m=particle.mass
        acceleration = particle.acc         
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
            # Ensure gamma_new does not imply v > c             
            gamma_new = max(gamma_new, 1)             
            energy_loss_threshold = 1e-10  # Adjust this value based on your requirements
            if abs(KE_initial - KE_final) > energy_loss_threshold:            
                # Solve for new velocity magnitude
                v_magnitude_new = self.c * np.sqrt(1 - 1 / (gamma_new**2))
                # Update velocity vector (maintaining direction)
                v_direction = v / np.linalg.norm(v)
                v_new = v_direction * v_magnitude_new
            else:
                # Energy loss is negligible, keep the initial velocity
                v_new = v                
        else:
            v_new = np.array([0.0, 0.0, 0.0])
            KE_final= m * self.c ** 2
        return v_new, delta_E, KE_final   
        
    ####################################    
    
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
    #########################################
    
    def calculate_electric_field(self, particle, r_retarded, r_retarded_mag,
                                 r_retarded_unit, retarded_velocity, retarded_acc):
        # Liénard-Wiechert potentials 
        beta = retarded_velocity / self.c
        gamma_ret = self.Gamma(beta)
        one_minus_beta_dot_r = 1 - np.dot(beta, r_retarded_unit)
        common_factor = particle.charge / (4 * np.pi * self.EPSILON_0 * (r_retarded_mag ** 2 ))#+ self.epsilon
        velocity_term = (r_retarded_unit - beta) / (gamma_ret **2 * one_minus_beta_dot_r ** 3)        
        acc_term = np.cross(r_retarded_unit, np.cross((r_retarded_unit -beta), retarded_acc/ self.c))/ (self.c * one_minus_beta_dot_r ** 3)
        return common_factor * (velocity_term+acc_term)#acc_term)#velocity_term np.array([0.0,0.0,0.0])
        
    #########################################    
    
    def calculate_magnetic_field(self, particle, r_retarded_mag, r_retarded_unit, retarded_velocity):
        # Biot-Savart law
        B_dir = np.cross(r_retarded_unit, retarded_velocity)
        B_dir_unit = B_dir / np.linalg.norm(B_dir) if np.linalg.norm(B_dir) > 0 else np.zeros(3)        
        dot_product = np.dot(r_retarded_unit, retarded_velocity)
        norm_product = np.linalg.norm(retarded_velocity) * np.linalg.norm(r_retarded_unit)
        theta = np.arccos(np.clip(dot_product / norm_product, -1, 1)) if norm_product > 0 else 0
        B_mag = (self.MU_0 / (4 * np.pi)) * (particle.charge * np.linalg.norm(retarded_velocity) * np.sin(theta)) / (r_retarded_mag ** 2+self.epsilon)
        return B_mag * B_dir_unit       
        
    #########################################
    
    def calculate_retardation(self,self_p, particle):
        r = self_p.position - particle.position 
        retarded_time = np.linalg.norm(r) / self.c
        return r, retarded_time
        
    #########################################
    
    def retarded_state(self, particle, retarded_time):
        accumulated_time = 0
        step_index = 0
        # Accumulate time steps from the end of self.T_traject backwards
        for i in range(len(particle.dt_traj) - 1, -1, -1):
            accumulated_time += particle.dt_traj[i]
            if accumulated_time >= retarded_time:
                # Found the index where the accumulated time just exceeds or matches the retarded time
                step_index = len(particle.dt_traj) - 1 - i
                break
        # Use the historical state from trajectory and past_vel if enough data is available
        if step_index < len(particle.pos_traj) :
            retarded_position = particle.pos_traj[-step_index - 1]
            retarded_velocity = particle.vel_traj[-step_index - 1]
            retarded_acc = particle.acc_traj[-step_index - 1]
        else:
            # Use the last available state from the historical data
            if len(particle.pos_traj) > 0:
                retarded_position =  particle.pos_traj[-1] 
                retarded_velocity = particle.vel_traj[-1]
                retarded_acc = particle.acc_traj[ - 1]
                remaining_time = retarded_time - accumulated_time
                dt_d =  particle.dt_traj[ - 1]  #  time step for RK4 calculation  
                # Step backward using RK4 until the desired retarded time is reached
                #this section need more work, I choose very long trajectory memory to
                #avoid that calculation, but it is easily doable. 
                #while remaining_time > 0:
                    #step_dt = min(dt_d, remaining_time)                     
                    #retarded_position = particle.position [-1]
                    #retarded_velocity = particle.velocity[-1]
                    #retarded_acc = particle.acc  [-1]                        
                    #retarded_position, retarded_velocity =  em_equations.rk4_step(-step_dt, retarded_position, 
                    #                                                     v_new, particle.force_method)
                    #remaining_time -= step_dt
            else:
                # If no historical data is available, use the current state as a fallback
                retarded_position = particle.position 
                retarded_velocity = particle.velocity
                retarded_acc = particle.acc                
        return retarded_position, retarded_velocity , retarded_acc  

    #########################################
    def calculate_retarded_distance(self, particle, retarded_position):
        """
        Calculate the vector difference, magnitude, and unit vector from 
        self.position to retarded_position.
    
        Parameters:
        - retarded_position: numpy array representing the retarded position.    
        Returns:
        - r_retarded: Vector difference from retarded_position to self.position.
        - r_retarded_mag: Magnitude of the vector difference.
        - r_retarded_unit: Unit vector in the direction from retarded_position to self.position.
        """
        r_retarded = particle.position - retarded_position
        r_retarded_mag = np.linalg.norm(r_retarded)
        if r_retarded_mag > 0:
            r_retarded_unit = r_retarded / r_retarded_mag
        else:
            r_retarded_unit = np.zeros_like(r_retarded)
        return r_retarded, r_retarded_mag, r_retarded_unit    
        
    #########################################               
    #########################################       
    
    def lorentz_transform_fields(self, E, B, v):
        E = np.array(E)
        B = np.array(B)
        v = np.array(v)
        if E.shape[0] != 3 or B.shape[0] != 3 or v.ndim != 1 or len(v) != 3:
            raise ValueError("Electric and magnetic fields must have the first dimension size 3, and velocity must be 1D with three elements.")
        gamma = self.Gamma(v) 
        v_norm = np.linalg.norm(v)                 
        # Expand the shapes of B and v to match E
        B_expanded = B.reshape(B.shape + (1,) * (E.ndim - 1))
        v_expanded = v.reshape(v.shape + (1,) * (E.ndim - 1))        
        # Lorentz transformations for electric and magnetic fields
        E_prime = gamma * (E + np.cross(v_expanded, B_expanded, axis=0)) - gamma**2 / (gamma + 1) * np.sum(v_expanded * E, axis=0, keepdims=True) * v_expanded / self.c**2
        B_prime = gamma * (B_expanded - np.cross(v_expanded, E, axis=0) / self.c**2) - gamma**2 / (gamma + 1) * np.sum(v_expanded * B_expanded, axis=0, keepdims=True) * v_expanded / self.c**2
        
        return E_prime, B_prime      

    #########################################
    def Gamma(self, v):        
        """
        Calculate the Lorentz factor for a given velocity.
        Parameters:
        - v: Velocity vector of the particle (numpy array).
        Returns:
        - Lorentz factor (gamma).
        """         
        v_copy = v.copy()
        return 1 / np.sqrt(1 - np.linalg.norm(v_copy)**2 / self.c**2)

    #########################################
    def limit_speed(self,v, limit= 0.9999999):
        """ 
        Limit the magnitude of the velocity vector to a maximum speed.
        Parameters:
        - v: Velocity vector of the particle (numpy array).
        - limit: Maximum allowed speed (default is the speed of light).
        Returns:
        - Adjusted velocity vector.
        """        
        v_copy = v.copy()
        speed = np.linalg.norm(v_copy)
        if speed > (self.c * limit): 
            return (v_copy / speed) * (limit * self.c)  
        return v_copy              
    
    def coulombs_law(self, q1, q2, r):
        # Implementation of Coulomb's law equation
        pass

    def electric_field(self, q, r):
        # Implementation of electric field equation
        pass

    def magnetic_field(self, i, r):
        # Implementation of magnetic field equation
        pass

    def lorentz_force(self, q, e, v, b):
        # Implementation of Lorentz force equation
        pass

    def faradays_law(self, n, phi, t):
        # Implementation of Faraday's law equation
        pass

    def amperes_law(self, i, l, b):
        # Implementation of Ampère's law equation
        pass

    # Add more equations as needed
    
    
    

 

 

 