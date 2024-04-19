import numpy as np
import pygame 
from physics import ElectromagneticEquations  
import PySimpleGUI as sg
import time 
import random
class VisualizationManager:
    def __init__(self, width, height, GRID_SPACING, num_particle, particle_manager, zoom_factor=1):
        self.WIDTH, self.HEIGHT = width, height
        self.GRID_SPACING = GRID_SPACING
        self.zoom_factor = zoom_factor

        self.epsilon = np.finfo(np.float64).eps
        self.zoom_center = [self.WIDTH // 2, self.HEIGHT // 2]
        self.ARROW_COLOR_E = (128, 128, 128)
        self.ARROW_COLOR_B = (128, 0, 0)
        #self.update_arrow_grid()  # Set up the initial arrow grid

        self.em_equations = ElectromagneticEquations()
        self.e_mass, self.p_mass = self.em_equations.e_mass, self.em_equations.p_mass
        self.e_charge = -self.em_equations.ELEMENTARY_CHARGE         
        self.p_charge = self.em_equations.ELEMENTARY_CHARGE
        self.num_particle = num_particle
        self.particle_manager = particle_manager
        pygame.init()
        
        self.fonts = {}
        self.x_in, self.y_in = np.meshgrid(range(0, self.WIDTH, self.GRID_SPACING), 
                                           range(0, self.HEIGHT,self.GRID_SPACING))   
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Particle Simulation")
        self.clock = pygame.time.Clock()
        self.electric_field_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)  

#############################################################################################
    def run_simulation(self, mass_mults, charge_mult, e_initial_velocity, p_initial_velocity,
                       ERROR_THRESHOLD, dt, B, Q_F, electric_field, magnetic_field):
        """
        Initialize and run the particle simulation.
        
        Parameters:
        - mass_mults: Multipliers for particle masses.
        - charge_mult: Multiplier for particle charges.
        - e_initial_velocity, p_initial_velocity: Initial velocities for electrons and protons.
        - ERROR_THRESHOLD: Error threshold for simulation accuracy.
        - dt: Time step for simulation updates.
        - B, Q_F: Magnetic field and charge force parameters.
        - electric_field, magnetic_field: Fields affecting particle dynamics.
        """
        
        # Setup the simulation GUI with necessary parameters
        self.setup_simulation_gui(B,mass_mults)        
        # Assign parameters to instance variables
        self.mass_mults= mass_mults
        self.mass_mults_sim =mass_mults
        self.charge_mult = charge_mult
        self.e_initial_velocity, self.p_initial_velocity = e_initial_velocity, p_initial_velocity
        self.dt, self.B, self.Q_F = dt, B, Q_F
        self.electric_field, self.magnetic_field = electric_field, magnetic_field
        self.allow_creation = True        
        # Define available simulation methods
        self.method_names = [
            ("boris_push", 1, 'B'), ("analytic", 2, 'A'),                                 
            ('analytic_rel', 3, 'AR'), ("boris_push", 4, 'B'),                        
            ("vay_algorithm", 5, 'VA'), ("rk4_step", 6, 'L4'),
            ("rk6_step", 7, 'L6'), ("relativ_intgrtr", 8, 'R'),
            ("vay_push", 9, 'VP'), ("Hamiltonian", 10, 'H')    ]    
        # Initialize simulation with the first particle
        self.add_initial_particle()        
        # Execute the main simulation loop
        self.run_main_loop()        
        # Cleanup resources and exit the simulation properly
        self.shutdown_simulation()

    def setup_simulation_gui(self, B, mass_mults):
        layout = [
            [sg.Text('Magnetic Field'), sg.Slider(range=(-4000, 4000), orientation='h', 
                                                  key='Magnetic Field', default_value=B, resolution=0.001)],
            [sg.Text('Electric Field'), sg.Slider(range=(-3000, 3000), orientation='h', 
                                                  key='Electric Field', default_value=0.00, resolution=0.001)],
            [sg.Text('Mass Factor'), sg.Slider(range=(1, 2e3), orientation='h', 
                                               key='Mass Factor', default_value=mass_mults, resolution=1)],
            [sg.Checkbox('Particle Creation', key='allow_creation', default=True)],
            [sg.Button('Update Simulation', key='update_simulation'),
             sg.Button('Keep One Particle', key='keep_one_particle')],
        ]
        self.window = sg.Window('Simulation Controls', layout)
        _, values = self.window.read(timeout=10)
        self.update_simulation_parameters(values)

    def run_main_loop(self):
        timeout = 10
        counter = 0
        running = True
        while running:
            start_time = time.time()
            event, values = self.window.read(timeout=0)
            if event == sg.WINDOW_CLOSED:
                running = False  # Set running to False to close both windows 
            if event == 'update_simulation':
                self.update_simulation_parameters(values)                 
            if event == 'keep_one_particle':
                self.keep_one_particle()            
            # Pygame event handling
            pygame_running, counter = self.handle_pygame_events(counter)     
            running = running and pygame_running  # Combine states
            if not running:
                break
            # Particle management mass_mults_sim
            self.particle_manager.update(self.electric_field, self.magnetic_field,
                                         self.Q_F_sim, self.B_sim, self.mass_mults_sim, self.zoom_factor)
            self.remove_escaping_particles()            
            # Visualization            
            counter = self.visualize(self.electric_field, self.magnetic_field, self.Q_F_sim, self.B_sim, counter)            
            # Check if timeout exceeded
            if time.time() - start_time > timeout:
                break            
            # Maintain desired frame rate
            self.clock.tick(240)

 
    
    def keep_one_particle(self):
        if self.particle_manager.particles:
            self.particle_manager.particles = [self.particle_manager.particles[0]]         
    
    def update_simulation_parameters(self, values):
        self.B_sim = values['Magnetic Field']
        self.Q_F_sim = values['Electric Field']
        self.mass_mults_sim = values['Mass Factor']
        self.allow_creation = values['allow_creation']
    

 

    def handle_pygame_events(self, counter):
        num_particles = len(self.particle_manager.particles)
        for event in pygame.event.get():            
            if event.type == pygame.QUIT:
                return False, counter  # Return a tuple with running=False and counter
            elif event.type == pygame.MOUSEWHEEL:
                self.handle_mousewheel_event(event)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    counter = self.handle_mousebuttondown_event( counter, num_particles )
        return True, counter  

    def handle_mousebuttondown_event(self, counter, num_particles):
        mouse_x, mouse_y = pygame.mouse.get_pos()
        zoom_x, zoom_y = self.zoom_center
        x = zoom_x + (mouse_x - zoom_x) / self.zoom_factor
        y = zoom_y + (mouse_y - zoom_y) / self.zoom_factor        
        if num_particles < self.num_particle and self.allow_creation:
            # Get the velocity_method of all existing particles
            used_methods = [particle.velocity_method for particle in self.particle_manager.particles]
            
            # Count the usage of each method
            method_usage = {}
            for method in used_methods:
                method_name = method.__name__
                if method_name in method_usage:
                    method_usage[method_name] += 1
                else:
                    method_usage[method_name] = 1
            
            # Sort the available methods by their usage count (prefer less used methods)
            sorted_methods = sorted(self.method_names, key=lambda method: method_usage.get(method[0], 0))
            
            # Choose the least used method
            method_name, thickness, letter = sorted_methods[0]
            push_method = getattr(self.em_equations, method_name)
            
            self.particle_manager.add_particle(
                np.array([x, y, 0.0]),
                np.array(self.e_initial_velocity),
                self.charge_mult * self.e_charge,
                self.mass_mults * self.e_mass,
                self.dt,
                self.em_equations.Lorentz,
                self.em_equations.radiation_reaction,
                push_method, 
                self.electric_field,
                self.magnetic_field, 
                letter            )
            print(f'Particle added, method: {method_name}')
        
        counter += 1
        return counter







    def add_initial_particle(self):
        self.particle_manager.add_particle(
            np.array([400.0, 301.0, 0.0]),
            self.p_initial_velocity,
            self.charge_mult * self.p_charge,
            self.mass_mults * self.p_mass, self.dt,
            self.em_equations.Lorentz,
            self.em_equations.radiation_reaction,
            self.em_equations.boris_push,
            self.electric_field, self.magnetic_field,  'p'  )  

    def visualize(self, electric_field, magnetic_field, Q_F, B, counter, initial_zoom=1.0):
        num_particles = len(self.particle_manager.particles)
        self.screen.fill((0, 0, 0))
        self.electric_field_surface.fill((0, 0, 0, 0))    
        E_field = electric_field((self.x_in, self.y_in, 0), Q_F)
        B_field = magnetic_field((self.x_in, self.y_in, 0), B)
        E_transformed, B_transformed = self.transform_fields(E_field, B_field, counter, num_particles)    
        self.draw_arrows(E_transformed, B_transformed)
        #print(counter , num_particles, counter , num_particles)
        self.draw(self.particle_manager.particles, self.screen, counter % num_particles)
        self.draw_particle_info(counter, num_particles)    
        self.screen.blit(self.electric_field_surface, (0, 0))
        pygame.display.flip()
        return counter    

#############################################################################################         
    def draw_arrows(self, E_trans, B_trans):
        def process_field(field, arrow_scale):
            magnitude = np.linalg.norm(field, axis=0)
            safe_magnitude = np.where(magnitude > self.epsilon, magnitude, np.full_like(magnitude, self.epsilon))
            direction = field / safe_magnitude  
            arrow_lengths = 20 #np.minimum(magnitude * arrow_scale / self.zoom_factor, 25)
            # Calculate end positions based on the new, zoomed grid
            arrow_end_x = self.x_in + arrow_lengths * direction[0]
            arrow_end_y = self.y_in + arrow_lengths * direction[1]
            start_pos = np.column_stack((self.x_in.ravel(), self.y_in.ravel()))
            end_pos = np.column_stack((arrow_end_x.ravel(), arrow_end_y.ravel()))
            return start_pos, end_pos    
        # draw arrows for both fields
        E_start_pos, E_end_pos = process_field(E_trans, arrow_scale=20)
        B_start_pos, B_end_pos = process_field(B_trans, arrow_scale=20)    
        for start, end in zip(E_start_pos, E_end_pos):
            pygame.draw.line(self.electric_field_surface, self.ARROW_COLOR_E, start.astype(int), end.astype(int), 1)    
        for start, end in zip(B_start_pos, B_end_pos):
            pygame.draw.line(self.electric_field_surface, self.ARROW_COLOR_B, start.astype(int), end.astype(int), 2)

    def update_arrow_grid(self):
        # Adjust grid spacing dynamically based on zoom factor
        dynamic_spacing = int(self.GRID_SPACING / self.zoom_factor)
        dynamic_spacing = max(10, dynamic_spacing)  # Ensure spacing does not get too small
    
        x_positions = range(0, self.WIDTH, dynamic_spacing)
        y_positions = range(0, self.HEIGHT, dynamic_spacing)
        self.x_in, self.y_in = np.meshgrid(x_positions, y_positions)
    
        # Call redraw function if necessary
        self.redraw_all()  # Make sure this method refreshes the visualization properly


    
    def draw(self, particles, screen , count ):
        """Main drawing function."""
        MAX_TIME_DILATION = 200
        MAX_VELOCITY = self.em_equations.c
        MAX_ENERGY_LOSS = 1e-4        
        self.draw_trajectory(particles, screen)         
        self.draw_particles(particles, screen)
        self.draw_scale_marker(screen, 1000) 
        max_particle = particles[count]
        velocity = np.linalg.norm(max_particle.velocity)
        gamma = max_particle.gamma
        energy_loss=max_particle.energy_loss/(max_particle.gamma*max_particle.mass*(self.em_equations.c**2)+ self.epsilon)    
        screen_width, screen_height = screen.get_size()
        bar_positions = { "V[c]": (screen_width - 100, 160 + (screen_height - 20) // 2),
            "Gamma": (screen_width - 50, 160 + (screen_height - 20) // 2),
            "Enrg Loss": (screen_width - 150, 160 + (screen_height - 20) // 2)  }    
        self.draw_bar(screen, "V[c]", velocity, MAX_VELOCITY, bar_positions["V[c]"] )
        self.draw_bar(screen, "Gamma", gamma, MAX_TIME_DILATION, bar_positions["Gamma"] )
        self.draw_bar(screen, "Enrg Loss", max(energy_loss, 0), MAX_ENERGY_LOSS, bar_positions["Enrg Loss"])    
        font_dt = self.get_font(24) 
        #font = pygame.font.Font(None, 24)
        value_text = font_dt.render(f"Δt = {max_particle.dt:.1e}", True, (254, 254, 254))
        value_text_rect = value_text.get_rect(centerx=50, y=10)
        screen.blit(value_text, value_text_rect)

    def draw_particles(self, particles, screen):
        """Draws all particles on the screen using preloaded fonts and optimized calculations."""
        # Use the preloaded font for particles
        font = self.get_font(16) #self.fonts[16]  # Assuming 16 is an appropriate size for particle labels    
        for particle in particles:
            if not particle.active:
                continue  # Skip inactive particles to reduce unnecessary processing    
            # Convert world coordinates to screen coordinates
            x_screen = (particle.position[0] - self.zoom_center[0]) * self.zoom_factor + self.WIDTH // 2
            y_screen = (particle.position[1] - self.zoom_center[1]) * self.zoom_factor + self.HEIGHT // 2    
            # Draw the particle as a circle on the screen
            pygame.draw.circle(screen, (250, 250, 250), (int(x_screen), int(y_screen)), particle.radius, 0)    
            # Render the particle's label (e.g., "e-", "p+") at its position
            text = font.render(particle.letter, True, particle.color)
            text_rect = text.get_rect(center=(int(x_screen), int(y_screen)))
            screen.blit(text, text_rect)  # Blit the text onto the screen    
 
    def remove_escaping_particles(self):
        # Early exit if there are no particles
        if not self.particle_manager.particles:
            return
        positions = np.array([particle.position for particle in self.particle_manager.particles])
        velocities = np.array([particle.velocity for particle in self.particle_manager.particles])    
        if positions.size == 0 or velocities.size == 0:
            return  # Exit if there are no particles to process
        valid_velocities = ~np.isnan(velocities).any(axis=1)
        max_distance = 2 * max(self.WIDTH, self.HEIGHT)
        within_bounds = np.linalg.norm(positions, axis=1) < max_distance
        valid_indices = valid_velocities & within_bounds
        self.particle_manager.particles = [particle for particle, valid in zip(self.particle_manager.particles, valid_indices) if valid ]

 
    def draw_particle_info(self,   counter, num_particles):
        part_in = counter % num_particles 
        if self.particle_manager.particles:
            particle = self.particle_manager.particles[part_in ]
            center_x, center_y = self.WIDTH - 200, self.HEIGHT - 100            
            pygame.draw.circle(self.screen, particle.color, (center_x, center_y), 10)     
            font_p = self.get_font(18) 
            text = font_p.render(particle.letter, True, (254, 254, 254))
            text_rect = text.get_rect(center=(center_x, center_y))
            self.screen.blit(text, text_rect)   
            
    def draw_trajectory(self, particles, screen):
        """Draws the trajectory of a single particle."""
        for particle in particles:
            if hasattr(particle, 'pos_traj') and len(particle.pos_traj) > 1:
                pos_traj_screen = [
                    [(coord - zoom) * self.zoom_factor + axis // 2 for coord, zoom, 
                     axis in zip(pos, self.zoom_center, [self.WIDTH, self.HEIGHT])]
                    for pos in particle.pos_traj ]
                pygame.draw.lines(screen, particle.color, False, pos_traj_screen)

    def draw_bar(self, screen, label, value, max_value, position, mode='log', color=(0, 255, 0)):
        """Draws a labeled bar for displaying metrics like velocity or time dilation."""
        bar_color = (255, 0, 0)  # Red
        text_color = (255, 255, 255)  # White
        bar_width, bar_height = 20, 100
        bar_x, bar_y = position
        if mode == 'log' and value > 0 and max_value > 0:
            log_value = np.log10(value + self.epsilon)
            log_max_value = np.log10(max_value + self.epsilon)
            filled_height = int(bar_height * log_value / log_max_value)
        else:
            filled_height = int(bar_height * (value / max_value))

        pygame.draw.circle(self.screen, (250,250,250), (self.WIDTH//2, self.HEIGHT//2), 1)     

        pygame.draw.rect(screen, (255, 255, 255), (bar_x, bar_y, bar_width, bar_height), 1)
        pygame.draw.rect(screen, color, (bar_x, bar_y + bar_height - filled_height, bar_width, filled_height))
        font_trj = self.get_font(18) 
        label_text = font_trj.render(label, True, text_color)
        value_text = font_trj.render(f"{value:.1e}", True, bar_color)
        label_text_rect = label_text.get_rect(centerx=bar_x + bar_width // 2, y=bar_y + bar_height + 5)
        value_text_rect = value_text.get_rect(centerx=bar_x + bar_width // 2, bottom=bar_y - 10)
        screen.blit(label_text, label_text_rect)
        screen.blit(value_text, value_text_rect)                
     
############################################################################################     
    def draw_scale_marker(self, screen, scale_length_mm):
        scale_length_pixels = scale_length_mm / 1000 * self.zoom_factor        
        if scale_length_pixels < 1:
            while scale_length_pixels < 1:
                scale_length_mm *= 10
                scale_length_pixels = scale_length_mm / 1000 * self.zoom_factor
        elif scale_length_pixels > 50:
            while scale_length_pixels > 50:
                scale_length_mm /= 10
                scale_length_pixels = scale_length_mm / 1000 * self.zoom_factor        
        scale_start = (self.WIDTH - 20, self.HEIGHT - 20)
        scale_end = (self.WIDTH - 20 - scale_length_pixels, self.HEIGHT - 20)        
        pygame.draw.line(screen, (255, 255, 255), scale_start, scale_end, 2)        
        font = self.get_font(16)
        scale_text = font.render(f"{scale_length_mm:.1e} mm", True, (255, 255, 255))
        scale_text_rect = scale_text.get_rect(midright=(self.WIDTH - 25 - scale_length_pixels, self.HEIGHT - 20))
        screen.blit(scale_text, scale_text_rect)                
 
#############################################################################################            

    
    def handle_mousewheel_event(self, event):
        mouse_x, mouse_y = pygame.mouse.get_pos()
        #mouse_x, mouse_y = mouse_x/self.zoom_factor, mouse_y/self.zoom_factor
        zoom_factor_change = 1.05 if event.y > 0 else 0.9
        old_zoom_factor = self.zoom_factor
        self.zoom_factor *= zoom_factor_change
        self.zoom_factor = max(1e0, min(self.zoom_factor, 1e31))    
        # Calculate the new zoom center based on the mouse position
        zoom_x, zoom_y = self.zoom_center
        new_zoom_x = mouse_x
        new_zoom_y = mouse_y
        self.zoom_center = [new_zoom_x, new_zoom_y]        
        
#############################################################################################         
    def transform_fields(self, E_field, B_field, counter, num_particle):
        #print(counter,num_particle)
        def normalize_field(field):
            field = np.where(np.isfinite(field), field, 0)
            max_value = np.max(field)
            return field / (max_value + self.epsilon) if max_value != 0 else np.zeros_like(field)        
        velocity = np.array([0.0, 0.0, 0.0])    
        if self.particle_manager.particles and len(self.particle_manager.particles) == num_particle:
            velocity = self.particle_manager.particles[counter % num_particle].velocity    
        E_transformed, B_transformed = self.em_equations.lorentz_transform_fields(E_field, B_field, velocity)        
        B_transformed = normalize_field(B_transformed)
        E_transformed = normalize_field(E_transformed)        
        return E_transformed, B_transformed
         
    def update_display(self):
        # Update the display with the latest drawing
        pygame.display.flip()
        self.clock.tick(240)

    def preload_fonts(self):
        # Define the font sizes you need
        font_sizes = [16, 18, 24]
        # Preload fonts and store them in the dictionary
        self.fonts = {size: pygame.font.Font(None, size) for size in font_sizes}
    
    def get_font(self, size):
        """Retrieve a font of the given size, caching it if not already loaded."""
        if size not in self.fonts:
            self.fonts[size] = pygame.font.Font(None, size)
        return self.fonts[size]        
            
    def shutdown_simulation(self):
        try:
            pygame.quit()
        except Exception as e:
            print(f"Error shutting down Pygame: {e}")
    
        try:
            if self.window:
                self.window.close()
        except Exception as e:
            print(f"Error closing PySimpleGUI window: {e}")


      