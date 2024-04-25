import numpy as np
import pygame 
from physics import ElectromagneticEquations  
import PySimpleGUI as sg
import time 
import random
import math

class VisualizationManager:
    def __init__(self, width, height, GRID_SPACING, num_particle, particle_manager, zoom_factor=1):
        self.WIDTH, self.HEIGHT = width, height
        self.GRID_SPACING = GRID_SPACING
        self.zoom_factor = zoom_factor
        self.x_in, self.y_in = np.meshgrid(range(0, self.WIDTH, self.GRID_SPACING), 
                                           range(0, self.HEIGHT,self.GRID_SPACING))   

        
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
        self.method_names = [("boris_push", 'B'), ('analytic_rel', 'AR') , 
                             ("rk4_step", 'L4'),("rk6_step", 'L6'), ("relativ_intgrtr", 'R'),
                             ("vay_push", 'VP'), ("Hamiltonian", 'H') ,("vay_algorithm", 'VA')]  
        self.fonts = {}
 
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Particle Simulation")
        self.clock = pygame.time.Clock()
        self.electric_field_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)  

#############################################################################################
    def run_simulation(self, mass_mults, charge_mult, e_initial_velocity, p_initial_velocity, 
                       dt, B_m, E_m, electric_field, magnetic_field):      
        # Setup the simulation GUI with necessary parameters
        self.setup_simulation_gui(B_m,E_m,mass_mults,charge_mult)        
        self.mass_mults= mass_mults
        self.mass_mults_sim =mass_mults
        self.charge_mult = charge_mult
        self.e_initial_velocity, self.p_initial_velocity = e_initial_velocity, p_initial_velocity
        self.dt, self.B_m, self.E_m = dt, B_m, E_m
        self.electric_field, self.magnetic_field = electric_field, magnetic_field
        self.allow_creation = True        
        self.add_initial_particle()        
        # Execute the main simulation loop
        self.run_main_loop()        
        # Cleanup resources and exit the simulation properly
        self.shutdown_simulation() 

    def setup_simulation_gui(self, B_m,E_m, mass_mults, charge_mult):
        method_options = [name for name, _ in self.method_names]
        layout = [[sg.Text('Magnetic Field'), sg.Slider(range=(-40, 40), orientation='h', 
                                                        key='Magnetic Field', default_value=B_m, resolution=0.001)],
                  [sg.Text('Electric Field'), sg.Slider(range=(-30, 30), orientation='h', 
                                                        key='Electric Field', default_value=E_m, resolution=0.001)],
                  [sg.Text('Mass Factor'), sg.Slider(range=(1, 200), orientation='h', 
                                                     key='Mass Factor', default_value=mass_mults, resolution=1)],
                  [sg.Text('Charge Factor'), sg.Slider(range=(-10000, 10000), orientation='h', 
                                                     key='Charge Factor', default_value=charge_mult, resolution=1)],                  
                  [sg.Text('Select Method'), sg.Combo(method_options, key='Selected Method', default_value='Hamiltonian')],
                  [sg.Checkbox('Particle Creation', key='allow_creation', default=True)],
                  [sg.Button('Update Simulation', key='update_simulation'),
                   sg.Button('Keep One Particle', key='keep_one_particle')],        ]
        self.window = sg.Window('Simulation Controls', layout)
        _, values = self.window.read(timeout=10)
        self.update_simulation_parameters(values)      
    
    def add_initial_particle(self):
        self.particle_manager.add_particle(
            np.array([402.0, 302.0, 0.0]),
            self.p_initial_velocity,
            self.charge_mult * self.p_charge,
            self.mass_mults * self.p_mass, self.dt,
            self.em_equations.Lorentz,
            self.em_equations.radiation_reaction,
            self.em_equations.boris_push,
            self.electric_field, self.magnetic_field,  'p'  )  
###################################################################################################        
    def run_main_loop(self):
        timeout = 10
        counter = 0
        running = True
        while running:
            start_time = time.time()  # Start timing
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
                                         self.E_m_sim, self.B_m_sim, 
                                         self.mass_mults_sim, self.charge_mults_sim, self.zoom_factor)
            self.remove_escaping_particles()            
            # Visualization            
            counter = self.visualize(self.electric_field, self.magnetic_field, 
                                     self.E_m_sim, self.B_m_sim, counter)            
            # Check if timeout exceeded
            if time.time() - start_time > timeout:
                break            
            # Maintain desired frame rate
            end_time = time.time()
            #iteration_duration = end_time - start_time            
            #print(iteration_duration)
            self.clock.tick(40) 
 
    def update_simulation_parameters(self, values):
        self.B_m_sim = values['Magnetic Field']
        self.E_m_sim = values['Electric Field']
        self.mass_mults_sim = values['Mass Factor']
        self.charge_mults_sim = values['Charge Factor']
        self.allow_creation = values['allow_creation']      
        self.selected_method = values['Selected Method']        

    def keep_one_particle(self):
        if self.particle_manager.particles:
            self.particle_manager.particles = [self.particle_manager.particles[0]]  
    
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

    def handle_mousewheel_event(self, event):
        mouse_screen_x, mouse_screen_y = pygame.mouse.get_pos()
        zoom_factor_change = 1.05 if event.y > 0 else 0.95
        self.zoom_factor *= zoom_factor_change
    
        world_x, world_y = self.screen_to_world(mouse_screen_x, mouse_screen_y, self.zoom_center, self.zoom_factor)
        self.zoom_factor = self.zoom_factor * zoom_factor_change
        new_mouse_screen_x, new_mouse_screen_y = self.world_to_screen(world_x, world_y, self.zoom_center, self.zoom_factor)
    
        screen_dx = mouse_screen_x - new_mouse_screen_x
        screen_dy = mouse_screen_y - new_mouse_screen_y
        self.zoom_center[0] += screen_dx
        self.zoom_center[1] += screen_dy    

    def handle_mousewheel_event1(self, event):
        mouse_screen_x, mouse_screen_y = pygame.mouse.get_pos()    
        # Apply zoom factor change
        zoom_factor_change = 1.05 if event.y > 0 else 0.95
        new_zoom_factor = self.zoom_factor * zoom_factor_change        
        # Convert screen coordinates to world coordinates using the current zoom center and factor
        world_x, world_y = self.screen_to_world(mouse_screen_x, mouse_screen_y, self.zoom_center, self.zoom_factor)        
        # Update the zoom factor
        self.zoom_factor = new_zoom_factor  
        #print('zoom_factor', self.zoom_factor)
        # Convert the world coordinates back to screen coordinates using the new zoom factor
        new_mouse_screen_x, new_mouse_screen_y = self.world_to_screen(world_x, world_y, self.zoom_center, self.zoom_factor)        
        # Calculate the difference between where the mouse was and where the point has moved on screen
        screen_dx = mouse_screen_x - new_mouse_screen_x
        screen_dy = mouse_screen_y - new_mouse_screen_y        
        # Shift the zoom center by the difference to keep the particle under the mouse stationary
        self.zoom_center[0] += screen_dx
        self.zoom_center[1] += screen_dy  
    
    def remove_escaping_particles(self):
        if not self.particle_manager.particles:
            return        
        positions = np.array([particle.position for particle in self.particle_manager.particles])
        velocities = np.array([np.squeeze(particle.velocity) for particle in self.particle_manager.particles])        
        # Calculate if any component of the velocity is NaN and invert the result
        valid_velocities = ~np.isnan(velocities).any(axis=1)        
        max_distance = 2 * max(self.WIDTH, self.HEIGHT)
        within_bounds = np.linalg.norm(positions, axis=1) < max_distance        
        # Combine conditions
        valid_indices = valid_velocities & within_bounds        
        # Initialize a new list for particles to keep
        new_particle_list = []
        for particle, valid in zip(self.particle_manager.particles, valid_indices):
            if valid:
                new_particle_list.append(particle)
            else:
                print(f"Removing particle: {particle.letter}")           
        self.particle_manager.particles = new_particle_list    
##########################################################################################################
    def visualize(self, electric_field, magnetic_field, E_m, B_m, counter, initial_zoom=1.0):
        num_particles = len(self.particle_manager.particles)
        self.screen.fill((0, 0, 0))
        self.electric_field_surface.fill((0, 0, 0, 0))
    
        # Visible dimensions based on zoom factor
        visible_width = self.WIDTH / self.zoom_factor
        visible_height = self.HEIGHT / self.zoom_factor
    
        # New grid spacing to maintain the same number of arrows
        num_cells_x = int(self.WIDTH / self.GRID_SPACING)
        num_cells_y = int(self.HEIGHT / self.GRID_SPACING)
        new_spacing_x = visible_width / num_cells_x
        new_spacing_y = visible_height / num_cells_y
    
        # Calculate start points for the grid to align with zoom center
        start_x = self.zoom_center[0] - visible_width / 2
        start_y = self.zoom_center[1] - visible_height / 2
    
        # Ensure start coordinates are within bounds
        start_x = max(0, start_x)
        start_y = max(0, start_y)
    
        # Generate the meshgrid for the current viewport
        x_in, y_in = np.meshgrid(
            np.linspace(start_x, start_x + visible_width, num_cells_x),
            np.linspace(start_y, start_y + visible_height, num_cells_y)
        )
    
        Z = np.zeros_like(x_in)
        positions = np.dstack((x_in, y_in, Z))
        E_field = electric_field(positions, E_m)
        B_field = magnetic_field(positions, B_m)
    
        # Transform and visualize fields
        if num_particles > 0:
            E_transformed, B_transformed = self.transform_fields(E_field, B_field, counter, num_particles)
        else:
            E_transformed, B_transformed = E_field, B_field  # Fallback if no particles
    
        self.draw_arrows(E_transformed, B_transformed)
        #print(counter , num_particles, counter , num_particles)
        self.draw(self.particle_manager.particles, self.screen, counter % num_particles)
        self.draw_particle_info(counter, num_particles)    
        self.screen.blit(self.electric_field_surface, (0, 0))
        pygame.display.flip()
        return counter  
 


    
    def visualize1(self, electric_field, magnetic_field, E_m, B_m, counter, initial_zoom=1.0):
        num_particles = len(self.particle_manager.particles)
        self.screen.fill((0, 0, 0))
        self.electric_field_surface.fill((0, 0, 0, 0))   
        effective_grid_spacing = max(1, int(self.GRID_SPACING / self.zoom_factor))
        self.x_in, self.y_in = np.meshgrid(
            np.arange(0, self.WIDTH, effective_grid_spacing),
            np.arange(0, self.HEIGHT, effective_grid_spacing)
        )

        
        Z = np.zeros_like(self.x_in)    
        # Stack X, Y, and Z to form a 3D array where each position vector is (x, y, z)
        positions = np.dstack((self.x_in, self.y_in, Z))
        E_field = electric_field(positions, E_m)
        B_field = magnetic_field(positions, B_m)
        E_transformed, B_transformed = self.transform_fields(E_field, B_field, counter, num_particles)    
        self.draw_arrows(E_transformed, B_transformed)
        #print(counter , num_particles, counter , num_particles)
        self.draw(self.particle_manager.particles, self.screen, counter % num_particles)
        self.draw_particle_info(counter, num_particles)    
        self.screen.blit(self.electric_field_surface, (0, 0))
        pygame.display.flip()
        return counter                              

    def handle_mousebuttondown_event(self, counter, num_particles):
        mouse_x, mouse_y = pygame.mouse.get_pos()
        zoom_x, zoom_y = self.zoom_center
        x, y = self.screen_to_world(mouse_x, mouse_y, (zoom_x, zoom_y), self.zoom_factor)        
        if num_particles < self.num_particle and self.allow_creation:
            method_name = self.selected_method
            push_method = getattr(self.em_equations, method_name)    
            self.particle_manager.add_particle(
                np.array([x, y, 0.0]),
                np.array(self.e_initial_velocity),
                self.charge_mult * self.e_charge,
                self.mass_mults * self.e_mass,
                self.dt,
                self.em_equations.Lorentz,  # Lorentz, Landau_Lifshitz
                self.em_equations.radiation_reaction,
                push_method, 
                self.electric_field,
                self.magnetic_field, 
                method_name[0:3]  # assuming method_names stored as (method_name, abbreviation)
            )
            print(f'Particle added, method: {method_name}')        
        counter += 1
        return counter    
        
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
        value_text = font_dt.render(f"Δt = {max_particle.dt:.1e}", True, (254, 254, 254))
        value_text_rect = value_text.get_rect(centerx=50, y=10)
        screen.blit(value_text, value_text_rect)
    
    def draw_arrows(self, E_trans, B_trans):
        def process_field(field, arrow_scale):
            # Calculate magnitudes and directions appropriately
            magnitude = np.linalg.norm(field, axis=-1)   
            safe_magnitude = np.where(magnitude > self.epsilon, magnitude, np.full_like(magnitude, self.epsilon))
            direction = field / safe_magnitude[..., np.newaxis]       
            # Calculate end positions based on the new, zoomed grid
            arrow_lengths = 20  # Static length for simplicity, can adjust based  
            arrow_end_x = self.x_in + arrow_lengths * direction[..., 0]   
            arrow_end_y = self.y_in + arrow_lengths * direction[..., 1]     
            start_pos = np.column_stack((self.x_in.ravel(), self.y_in.ravel()))
            end_pos = np.column_stack((arrow_end_x.ravel(), arrow_end_y.ravel()))    
            return start_pos, end_pos        
        # Process and draw arrows for both electric and magnetic fields
        E_start_pos, E_end_pos = process_field(E_trans, arrow_scale=20)
        B_start_pos, B_end_pos = process_field(B_trans, arrow_scale=20)        
        for start, end in zip(E_start_pos, E_end_pos):
            pygame.draw.line(self.electric_field_surface, self.ARROW_COLOR_E, start.astype(int), end.astype(int), 1)    
        for start, end in zip(B_start_pos, B_end_pos):
            pygame.draw.line(self.electric_field_surface, self.ARROW_COLOR_B, start.astype(int), end.astype(int), 2)  

    def draw_particles(self, particles, screen):
        """Draws all particles on the screen using preloaded fonts and optimized calculations."""
        font = self.get_font(16)    
        for particle in particles:
            if not particle.active:
                continue  # Skip inactive particles                
            try:
                x_screen = (particle.position[0] - self.zoom_center[0]) * self.zoom_factor + self.WIDTH // 2
                y_screen = (particle.position[1] - self.zoom_center[1]) * self.zoom_factor + self.HEIGHT // 2
            except TypeError as e:
                print(f"Invalid value for particle position encountered: {e}")
                print(f"Particle letter: {particle.letter}")
                continue  # Skip drawing this particle            
            # Check if the computed screen coordinates are valid numbers
            if x_screen is None or not isinstance(x_screen, (int, float)):
                print(f"Invalid x_screen value: {x_screen} for particle with letter: {particle.letter}")
                return x_screen, particle.letter  # Return the value and the particle's letter    
            # Draw the particle as a circle on the screen
            pygame.draw.circle(screen, (250, 250, 250), (int(x_screen), int(y_screen)), particle.radius, 0)            
            text = font.render(particle.letter, True, particle.color)
            text_rect = text.get_rect(center=(int(x_screen), int(y_screen)))
            screen.blit(text, text_rect)  # Blit the text onto the screen     
 
#############################################################################################                


    def update_arrow_grid(self):
        # Adjust grid spacing dynamically based on zoom factor
        dynamic_spacing = int(self.GRID_SPACING / self.zoom_factor)
        dynamic_spacing = max(10, dynamic_spacing)  # Ensure spacing does not get too small    
        x_positions = range(0, self.WIDTH, dynamic_spacing)
        y_positions = range(0, self.HEIGHT, dynamic_spacing)
        self.x_in, self.y_in = np.meshgrid(x_positions, y_positions)    
        self.redraw_all()  # Make sure this method refreshes the visualization properly    

    def draw_particle_info(self, counter, num_particles):
        part_in = counter % num_particles 
        if self.particle_manager.particles:
            particle = self.particle_manager.particles[part_in]
            center_x, center_y = self.WIDTH - 200, self.HEIGHT - 100            
            pygame.draw.circle(self.screen, particle.color, (center_x, center_y), 10)            
            # Draw particle info
            font_p = self.get_font(18) 
            particle_text = font_p.render(particle.letter, True, (254, 254, 254))
            particle_text_rect = particle_text.get_rect(center=(center_x, center_y))
            self.screen.blit(particle_text, particle_text_rect)    
        # Draw zoom factor at the top-left corner
        zoom_text = f"Zoom: {self.zoom_factor:.1e}"
        font_zoom = self.get_font(18)  # You can adjust the font size as needed
        zoom_surface = font_zoom.render(zoom_text, True, (254, 254, 254))
        zoom_rect = zoom_surface.get_rect(topleft=(10, 30))  # Position at the top-left corner        
        self.screen.blit(zoom_surface, zoom_rect)
        zoom_text = f"Proposed Δt: {1e-9 * (1 / self.zoom_factor):.1e}"
        font_zoom = self.get_font(18)  # You can adjust the font size as needed
        zoom_surface = font_zoom.render(zoom_text, True, (254, 254, 254))
        zoom_rect = zoom_surface.get_rect(topleft=(10, 50))  # Position at the top-left corner        
        self.screen.blit(zoom_surface, zoom_rect)        
            
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
            if math.isnan(value) or math.isnan(max_value) or max_value == 0:
                filled_height = 0  # Default to zero or some other appropriate error handling
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
     
    def screen_to_world(self, screen_x, screen_y, zoom_center, zoom_factor):
        # Convert screen coordinates to world coordinates
        world_x = (screen_x - zoom_center[0]) / zoom_factor + zoom_center[0]
        world_y = (screen_y - zoom_center[1]) / zoom_factor + zoom_center[1]
        return world_x, world_y

    def world_to_screen(self, world_x, world_y, zoom_center, zoom_factor):
        # Convert world coordinates back to screen coordinates
        screen_x = (world_x - zoom_center[0]) * zoom_factor + zoom_center[0]
        screen_y = (world_y - zoom_center[1]) * zoom_factor + zoom_center[1]
        return screen_x, screen_y                 
 
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
        
#############################################################################################               
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

      