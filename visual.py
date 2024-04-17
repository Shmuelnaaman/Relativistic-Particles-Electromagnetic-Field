import numpy as np
import pygame 
from physics import ElectromagneticEquations  
import PySimpleGUI as sg
import time 
class VisualizationManager:
    def __init__(self,width, height, GRID_SPACING, num_particle, particle_manager):
        self.WIDTH, self.HEIGHT = width, height
        self.GRID_SPACING = GRID_SPACING
        self.zoom_factor = 1.0
        self.epsilon = np.finfo(np.float64).eps 
        self.zoom_center = [self.WIDTH // 2, self.HEIGHT // 2]
        self.ARROW_COLOR_E = (128, 128, 128)
        self.ARROW_COLOR_B = (128, 0, 0)
        self.em_equations = ElectromagneticEquations()
        self.e_mass, self.p_mass = self.em_equations.e_mass, self.em_equations.p_mass
        self.e_charge, self.p_charge = -self.em_equations.ELEMENTARY_CHARGE, self.em_equations.ELEMENTARY_CHARGE
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
    
        layout = [[sg.Text('Magnetic Field'),
                   sg.Slider(range=(0, 5), orientation='h', key='B', default_value=0.05, resolution=0.001)],
                  [sg.Text('Electric Field'),
                   sg.Slider(range=(-1, 1), orientation='h', key='Q_F', default_value=0.00, resolution=0.001)],
                  [sg.Text('Mass Factor'),
                   sg.Slider(range=(1, 1e6), orientation='h', key='M_F', default_value=1e3, resolution=1000)],
                  [sg.Checkbox('Particle Creation', key='allow_creation', default=True)],
                  [sg.Checkbox('Update Simulation', key='update_simulation', default=False)]]
        window = sg.Window('Simulation Controls', layout)
    
        method_names = [("analytic", 1, 'A'),
                        ("boris_push", 2, 'B'),                        
                        ("vay_algorithm", 7, 'V_a'),
                        ("rk4_step", 3, 'L-4'),
                        ("rk6_step", 4, 'L-6'),
                        ("relativ_intgrtr", 6, 'R'),("vay_push", 5, 'V_p'),
                        ("Hamiltonian", 8, 'H')  ] # ("vay_push", 5, 'V_p'),
        event, values = window.read(timeout=10)
        B = values['B']
        Q_F = values['Q_F']
        mass_mults1 = values['M_F']
        B_prev, Q_F_prev, mass_mults1_prev = B, Q_F, mass_mults1
        update_simulation_prev = False

        self.add_initial_particle(p_initial_velocity, charge_mult, mass_mults, dt,
                                  electric_field, magnetic_field)
        timeout = 10  # 
        counter = 0
        
        running = True
        while running:
            start_time = time.time()
            event, values = window.read(timeout=10)
            if event == sg.WINDOW_CLOSED:
                running = False
                window.close()  # Close the control window when explicitly closed by the user
                break  

            update_simulation = values['update_simulation']
            if update_simulation:
                B = values['B']
                Q_F = values['Q_F']
                mass_mults1 = values['M_F']
            else:
                B, Q_F, mass_mults1 = B_prev, Q_F_prev, mass_mults1_prev    
            allow_creation = values['allow_creation']        
            
            running, counter = self.handle_pygame_events(e_initial_velocity, charge_mult, mass_mults, dt,
                                                         electric_field, magnetic_field, method_names, counter,
                                                         allow_creation)
            
            if not running:
                break
 
                        
            self.particle_manager.update(electric_field, magnetic_field, Q_F, B, mass_mults)
            self.remove_escaping_particles()     
            
            num_particles = len(self.particle_manager.particles)
            
            counter = self.visualize(electric_field, magnetic_field, Q_F, B, counter, num_particles)
            
            if time.time() - start_time > timeout:
                break 
            event, values = window.read(timeout=0) 
            self.clock.tick(240)    

            if update_simulation:
                B_prev, Q_F_prev, mass_mults1_prev = B, Q_F, mass_mults1
                update_simulation_prev = True
            else:
                update_simulation_prev = False
        pygame.quit()        
        window.close()
############################################################################################# 
    
    def handle_pygame_events(self, e_initial_velocity, charge_mult, mass_mults, dt,
                             electric_field, magnetic_field, method_names, counter, allow_creation):
        num_particles = len(self.particle_manager.particles)
        for event in pygame.event.get():            
            if event.type == pygame.QUIT:
                return False, counter  # Return a tuple with running=False and counter
            elif event.type == pygame.MOUSEWHEEL:
                self.handle_mousewheel_event(event)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    counter = self.handle_mousebuttondown_event(e_initial_velocity, charge_mult, mass_mults, dt,
                                                                electric_field, magnetic_field, method_names,
                                                                counter, num_particles, allow_creation)
        return True, counter

    def handle_mousebuttondown_event(self, e_initial_velocity, charge_mult, mass_mults, dt,
                                     electric_field, magnetic_field, method_names, counter, num_particles, allow_creation):
        mouse_x, mouse_y = pygame.mouse.get_pos()
        zoom_x, zoom_y = self.zoom_center
        x = zoom_x + (mouse_x - zoom_x) / self.zoom_factor
        y = zoom_y + (mouse_y - zoom_y) / self.zoom_factor    
        if num_particles < self.num_particle   and allow_creation:
            method_name, thickness, letter = method_names[num_particles % len(method_names)]
            push_method = getattr(self.em_equations, method_name)
            self.particle_manager.add_particle(
                np.array([x, y, 0.0]),
                np.array(e_initial_velocity),
                charge_mult * self.e_charge,
                mass_mults * self.e_mass, dt,
                self.em_equations.Landau_Lifshitz,
                self.em_equations.radiation_reaction,
                push_method, electric_field,
                magnetic_field, letter  )
            print(method_name)
            self.particle_manager.particles[num_particles].thickness = thickness
        counter += 1
        return counter   
        
    def remove_escaping_particles(self):
        # Early exit if there are no particles
        if not self.particle_manager.particles:
            return
    
        # Extract positions and velocities into numpy arrays
        positions = np.array([particle.position for particle in self.particle_manager.particles])
        velocities = np.array([particle.velocity for particle in self.particle_manager.particles])
    
        # Check for non-NaN velocities (assumes velocity is a numpy array for each particle)
        valid_velocities = ~np.isnan(velocities).any(axis=1)
    
        # Compute distances from the origin (or another reference point)
        # Assumes positions are [x, y, z] and we use only x, y for distance computation
        distances = np.linalg.norm(positions[:, :2], axis=1)
        max_distance = 2 * max(self.WIDTH, self.HEIGHT)
        within_bounds = distances <= max_distance
    
        # Combine conditions: valid velocities and within boundary limits
        valid_indices = valid_velocities & within_bounds
    
        # Filter particles based on combined conditions using list comprehension
        self.particle_manager.particles = [particle for particle, valid in zip(self.particle_manager.particles, valid_indices) if valid]


    def visualize(self, electric_field, magnetic_field, Q_F, B, counter, num_particles):
        self.screen.fill((0, 0, 0))
        self.electric_field_surface.fill((0, 0, 0, 0))    
        E_field = electric_field((self.x_in, self.y_in, 0), Q_F)
        B_field = magnetic_field((self.x_in, self.y_in, 0), B)
        E_transformed, B_transformed = self.transform_fields(E_field, B_field, counter, num_particles)    
        self.draw_arrows(E_transformed, B_transformed)
        self.draw(self.particle_manager.particles, self.screen, counter % num_particles)
        self.draw_particle_info(counter, num_particles)    
        self.screen.blit(self.electric_field_surface, (0, 0))
        pygame.display.flip()
        return counter
     
############################################################################################
    #            text onto the screen

    def draw(self, particles, screen , count ):
        """Main drawing function."""
        MAX_TIME_DILATION = 200
        MAX_VELOCITY = self.em_equations.c
        MAX_ENERGY_LOSS = 1e-4        
        self.draw_trajectory(particles, screen)         
        self.draw_particles(particles, screen)
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
        
    def draw_arrows(self, E_trans, B_trans):
        def process_field(field, arrow_scale):
            magnitude = np.linalg.norm(field, axis=0)
            safe_magnitude = np.where(magnitude > self.epsilon, magnitude, np.full_like(magnitude, self.epsilon))
            direction = field / safe_magnitude  # Ensure no division by zero
            arrow_lengths = np.minimum(magnitude * arrow_scale, 25)  # Adjust scale for visibility
            arrow_end_x = self.x_in + arrow_lengths * direction[0]
            arrow_end_y = self.y_in + arrow_lengths * direction[1]
            start_pos = np.column_stack((self.x_in.ravel(), self.y_in.ravel()))
            end_pos = np.column_stack((arrow_end_x.ravel(), arrow_end_y.ravel()))
            return start_pos, end_pos    
        def apply_zoom(pos, constant):
            return (pos - self.zoom_center) * self.zoom_factor + constant            
        constant = np.array([self.WIDTH // 2, self.HEIGHT // 2])
        E_start_pos, E_end_pos = process_field(E_trans, arrow_scale=20)  # Use a noticeable scale
        B_start_pos, B_end_pos = process_field(B_trans, arrow_scale=20)    
        E_start_pos, E_end_pos = apply_zoom(E_start_pos,  constant), apply_zoom(E_end_pos, constant)
        B_start_pos, B_end_pos = apply_zoom(B_start_pos, constant), apply_zoom(B_end_pos, constant)    
        for start, end in zip(E_start_pos, E_end_pos):
            pygame.draw.line(self.electric_field_surface, (128, 128, 128), start.astype(int), end.astype(int), 1)      
        for start, end in zip(B_start_pos, B_end_pos):
            pygame.draw.line(self.electric_field_surface, (255, 0, 0), start.astype(int), end.astype(int), 2)  

    def draw_particles(self, particles, screen):
        """Draws all particles on the screen."""    
        font = pygame.font.Font(None, 16)  # Create a font object just once, outside the loop
        for particle in particles:
            if not particle.active:
                continue  # Skip drawing inactive particles
            x, y, _ = particle.position
            x_screen = (x - self.zoom_center[0]) * self.zoom_factor + self.WIDTH // 2
            y_screen = (y - self.zoom_center[1]) * self.zoom_factor + self.HEIGHT // 2                
            pygame.draw.circle(screen, (120, 120, 0), (int(x_screen), int(y_screen)), particle.radius, 0)
            text = font.render(particle.letter, True, particle.color)  # Render the letter as text
            text_rect = text.get_rect(center=(int(x_screen), int(y_screen)))  
            screen.blit(text, text_rect)  # Blit the text onto the screen
        
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
        pygame.draw.rect(screen, (255, 255, 255), (bar_x, bar_y, bar_width, bar_height), 1)
        pygame.draw.rect(screen, color, (bar_x, bar_y + bar_height - filled_height, bar_width, filled_height))
        font_trj = self.get_font(18) 
        label_text = font_trj.render(label, True, text_color)
        value_text = font_trj.render(f"{value:.1e}", True, bar_color)
        label_text_rect = label_text.get_rect(centerx=bar_x + bar_width // 2, y=bar_y + bar_height + 5)
        value_text_rect = value_text.get_rect(centerx=bar_x + bar_width // 2, bottom=bar_y - 10)
        screen.blit(label_text, label_text_rect)
        screen.blit(value_text, value_text_rect)  
#############################################################################################        t    
# Add particles ext, text_rect)     

    def add_initial_particle(self, p_initial_velocity, charge_mult, mass_mults, dt,
                             electric_field, magnetic_field):
        self.particle_manager.add_particle(
            np.array([401.0, 300.0, 0.0]),
            p_initial_velocity,
            charge_mult * self.p_charge,
            mass_mults * self.p_mass, dt,
            self.em_equations.Lorentz,
            self.em_equations.radiation_reaction,
            self.em_equations.boris_push,
            electric_field, magnetic_field,  'p'  )
           

        
    # Zoom     
    
    def handle_mousewheel_event(self, event):
        mouse_x, mouse_y = pygame.mouse.get_pos()
        zoom_factor_change = 1.1 if event.y > 0 else 0.9
        old_zoom_factor = self.zoom_factor
        self.zoom_factor *= zoom_factor_change
        self.zoom_factor = max(0.1, min(self.zoom_factor, 1e6))    
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

    def quit(self):

                pygame.quit()
    def get_font(self, size):
        """Retrieve a font of the given size, caching it if not already loaded."""
        if size not in self.fonts:
            self.fonts[size] = pygame.font.Font(None, size)
        return self.fonts[size]        


      