% TIME_DRIVE  Sweep dt and drive dt_chk to measure temporal convergence.
% Save as time_drive.m and run from MATLAB path where dt_chk.m and helpers exist.

clearvars -except hdr semhat zwgll interp_mat; close all;
% initialize accumulator arrays
ktrial = 0;
kN = [];
kdt = [];
kem = [];
ktm = [];

% choose time-step vector (dt from coarse to fine)
dts = logspace(-3, -3.2, 3);   % example: [1e-3, 1e-3.666..., 1e-4.333..., 1e-5]
for i = 1:length(dts)
    dt = dts(i);
    % set nsteps so total time ~ 2*pi
    nsteps = round(2*pi / dt);
    % call dt_chk (function) and collect results
    [kN,kdt,kem,ktm,ktrial,X,Y,Er] = dt_chk(dt, nsteps, kN, kdt, kem, ktm, ktrial);
end

% Plot temporal convergence (error vs dt)
figure;
loglog(kdt, kem, 'r.', kdt, kem, 'k--', 'MarkerSize', 12);
axis square; grid on;

% Reverse x-axis so smaller dt is to the RIGHT
set(gca, 'XDir', 'reverse');

xlabel('\Delta t', 'Interpreter', 'none');
ylabel('||u - u_{exact}||_\infty', 'Interpreter', 'none');
title('Temporal Convergence (RK4)');

% Fit a line to log-log data to estimate observed order
ok = kem > 0 & kdt > 0;
if sum(ok) >= 2
    p = polyfit(log10(kdt(ok)), log10(kem(ok)), 1);
    slope = p(1);
    hold on;

    txt = sprintf('fit slope = %.2f', slope);
    xpos = 10^(mean(log10(kdt(ok))));
    ypos = 10^(mean(log10(kem(ok))));
    text(xpos, ypos, txt, ...
        'HorizontalAlignment', 'center', 'BackgroundColor', 'w');
end

% optionally print table to console
disp(' i   N    dt        err       t_final');
for i = 1:length(kN)
    fprintf('%2d  %3d  %8.3e  %8.3e  %8.3e\n', i, kN(i), kdt(i), kem(i), ktm(i));
end


figure;
mesh(X,Y,Er);

drawnow;
