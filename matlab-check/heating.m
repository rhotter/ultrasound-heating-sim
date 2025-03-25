% load the average intensity data
average_intensity = readNPY('../data/average_intensity.npy');
blood_perfusion_rate = readNPY('../data/medium/blood_perfusion_rate.npy');
density = readNPY('../data/medium/density.npy');
thermal_conductivity = readNPY('../data/medium/thermal_conductivity.npy');
specific_heat = readNPY('../data/medium/specific_heat.npy');
absorption_coefficient = readNPY('../data/medium/absorption_coefficient.npy');

pitch = 208e-6; % [m]

%%
% get grid dimensions from intensity array
[Nx, Ny, Nz] = size(average_intensity);

% create the computational grid
kgrid = kWaveGrid(Nx, pitch, Ny, pitch, Nz, pitch);

% set the background temperature and heating term
source.Q = 2 * absorption_coefficient .* average_intensity;
source.T0 = 37;

% % define medium properties related to diffusion
% medium.density              = 1020;     % [kg/m^3]
% medium.thermal_conductivity = 0.5;      % [W/(m.K)]
% medium.specific_heat        = 3600;     % [J/(kg.K)]

% new props
medium.perfusion_coeff = blood_perfusion_rate;
medium.density = density;
medium.thermal_conductivity = thermal_conductivity;
medium.specific_heat = specific_heat;
medium.blood_ambient_temperature = 37;

% create kWaveDiffusion object
kdiff = kWaveDiffusion(kgrid, medium, source, {'PlotSim', false, 'PlotFreq', 1000000000});

%%
% set source on time and off time
on_time  = 1000;  % [s]
off_time = 20;  % [s]

% set time step sizes
dt = 0.1;     % simulation time step [s]
dt_report = 1.0;  % reporting time step [s]

% Initialize arrays to store data for final plotting
time_points = [];
max_temps = [];

% Calculate steps
total_sim_time = on_time;
total_steps = round(total_sim_time / dt);

% Create figure for real-time plotting
figure('Name', 'Real-time Temperature Evolution');
h_plot = plot(NaN, NaN, 'b-', 'LineWidth', 1.5);
xlabel('Time [s]');
ylabel('Maximum Temperature [°C]');
title('Temperature Evolution Over Time');
grid on;

% take time steps
for step = 0:total_steps
    % Take one simulation step
    kdiff.takeTimeStep(1, dt);
    
    % Only record and print at reporting intervals
    current_time = step * dt;
    current_max_temp = max(kdiff.T(:));
    time_points(end+1) = current_time;
    max_temps(end+1) = current_max_temp;
    
    % Update the plot
    set(h_plot, 'XData', time_points, 'YData', max_temps);
    
    % Dynamically adjust both axes with padding
    temp_range = max_temps(end) - source.T0;
    
    % Add padding to axes, ensuring minimum ranges
    ylim([source.T0 - 0.1*temp_range, max_temps(end) + 0.1*temp_range]);
    
    % Ensure x-axis always has a valid range
    min_x_range = dt * 10;  % minimum range is 10 time steps
    current_x_range = max(current_time, min_x_range);
    xlim([0, current_x_range * 1.1]);  % add 10% padding
    
    drawnow;
    
    % Print current max temperature
    fprintf('Time: %.1f s, Max Temperature: %.2f °C\n', current_time, current_max_temp);
end

% store the final temperature field
T1 = kdiff.T;

% The final temperature evolution plot is already shown in real-time

%% 
% plot the temperature fields
% plot the temperature after heating
figure;
T1_slice = squeeze(T1(:,Ny/2,:));
imagesc(kgrid.y_vec * 1e3, kgrid.x_vec * 1e3, T1_slice);
h = colorbar;
xlabel(h, '[degC]');
ylabel('x-position [mm]');
xlabel('y-position [mm]');
axis image;
title('Temperature After Heating');

% % plot the temperature after cooling
% figure;
% T2_slice = squeeze(T2(:,Ny/2,:));
% imagesc(kgrid.y_vec * 1e3, kgrid.x_vec * 1e3, T2_slice);
% h = colorbar;
% xlabel(h, '[degC]');
% ylabel('x-position [mm]');
% xlabel('y-position [mm]');
% axis image;
% title('Temperature After Cooling');

%%
% Plot all thermal parameters in a grid
figure('Position', [100 100 1200 800]);

% Thermal conductivity
subplot(2,3,1);
k_slice = squeeze(thermal_conductivity(:,Ny/2,:));
imagesc(kgrid.y_vec * 1e3, kgrid.x_vec * 1e3, k_slice);
h = colorbar;
xlabel(h, '[W/(m·K)]');
ylabel('x-position [mm]');
xlabel('y-position [mm]');
title('Thermal Conductivity');

% Density
subplot(2,3,2);
rho_slice = squeeze(density(:,Ny/2,:));
imagesc(kgrid.y_vec * 1e3, kgrid.x_vec * 1e3, rho_slice);
h = colorbar;
xlabel(h, '[kg/m^3]');
ylabel('x-position [mm]');
xlabel('y-position [mm]');
title('Density');

% Specific heat
subplot(2,3,3);
c_slice = squeeze(specific_heat(:,Ny/2,:));
imagesc(kgrid.y_vec * 1e3, kgrid.x_vec * 1e3, c_slice);
h = colorbar;
xlabel(h, '[J/(kg·K)]');
ylabel('x-position [mm]');
xlabel('y-position [mm]');
title('Specific Heat');

% Blood perfusion
subplot(2,3,4);
w_slice = squeeze(blood_perfusion_rate(:,Ny/2,:));
imagesc(kgrid.y_vec * 1e3, kgrid.x_vec * 1e3, w_slice);
h = colorbar;
xlabel(h, '[1/s]');
ylabel('x-position [mm]');
xlabel('y-position [mm]');
title('Blood Perfusion Rate');

% plot average intensity
subplot(2,3,5);
average_intensity_slice = squeeze(average_intensity(:,Ny/2,:));
imagesc(kgrid.y_vec * 1e3, kgrid.x_vec * 1e3, average_intensity_slice);
h = colorbar;
xlabel(h, '[W/m^2]');
ylabel('x-position [mm]');
xlabel('y-position [mm]');
title('Average Intensity');

% plot heating source
subplot(2,3,6);
heating_source_slice = squeeze(source.Q(:,Ny/2,:));
imagesc(kgrid.y_vec * 1e3, kgrid.x_vec * 1e3, heating_source_slice);
h = colorbar;
xlabel(h, '[W/m^3]');
ylabel('x-position [mm]');
xlabel('y-position [mm]');
title('Heating Source');



% Make all plots have same axis scaling
for i = 1:6
    subplot(2,3,i);
    axis image;
end