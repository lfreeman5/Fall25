function [kN,kdt,kem,ktm,ktrial] = dt_chk(dt,nstep,kN,kdt,kem,ktm,ktrial)
% DT_CHK  Run one RK4 experiment for 2D advection and append results.
%   [kN,kdt,kem,ktm,ktrial] = dt_chk(dt,nstep,kN,kdt,kem,ktm,ktrial)
%
% Inputs:
%   dt      - time step
%   nstep   - number of steps (so total time ~ nstep*dt)
%   kN,kdt,kem,ktm,ktrial - arrays and counter to append results to
%
% Outputs:
%   updated kN,kdt,kem,ktm,ktrial
%
% Notes:
%   - Requires functions/files: hdr, semhat, zwgll, interp_mat
%   - This enforces boundary conditions (Mask) at every RK4 stage.

% Close figures/prepare header as original script did
close;
hdr; hold off;

% Problem setup (kept as in original)
N = 120;

[Ah,Bh,Ch,Dh,z,w] = semhat(N); Ih = speye(N+1);

Rx = Ih(2:end-1,:);
Ry = Ih(2:end-1,:);

Mask = zeros(N+1,N+1); Mask(2:end-1,2:end-1) = 1;

% interpolation to midpoints
[z1,w1] = zwgll(N-1);
J1 = interp_mat(z1,z); % interpolate to GLL "midpoints"

[X,Y] = ndgrid(z,z);

% velocity field and midpoint interpolation
Cx = -Y;   Cxm = J1*Cx*J1';
Cy =  X;   Cym = J1*Cy*J1';

% compute midpoint cell widths and scaled derivative-like quantities
dX = diff(X);  dX = dX * J1';         dUdx = Cxm ./ dX;
dY = diff(Y'); dY = dY'; dY = J1 * dY; dUdy = Cym ./ dY;

% set number of steps
nsteps = nstep;

% initial condition: Gaussian centered at (X0,Y0)
X0 = 0.5; Y0 = 0.0; delta = 0.1;
x = X - X0; y = Y - Y0; arg = -(x.*x + y.*y) / (delta^2);
U0 = exp(arg);

% enforce Dirichlet-like boundary (zero outside interior) on initial cond
U = Mask .* U0;

% RK4 constants
dt2 = dt / 2;
dt6 = dt / 6;

emax = 0;

% time-stepping
for istep = 1:nsteps
    time = dt * istep;

    % k1
    k1 = -Mask .* ( Cx .* (Dh * U) + Cy .* (U * Dh') );
    U1 = Mask .* (U + dt2 * k1);

    % k2
    k2 = -Mask .* ( Cx .* (Dh * U1) + Cy .* (U1 * Dh') );
    U2 = Mask .* (U + dt2 * k2);

    % k3
    k3 = -Mask .* ( Cx .* (Dh * U2) + Cy .* (U2 * Dh') );
    U3 = Mask .* (U + dt * k3);

    % k4
    k4 = -Mask .* ( Cx .* (Dh * U3) + Cy .* (U3 * Dh') );

    % update and re-enforce boundary mask
    U = Mask .* ( U + dt6 * (k1 + 2*(k2 + k3) + k4) );

    % error evaluation periodically (every 20 steps)
    if mod(istep,20) == 0
       c = cos(time); s = sin(time);
       X0t = c * X0 - s * Y0;
       Y0t = s * X0 + c * Y0;
       x = X - X0t; y = Y - Y0t; arg = -(x.*x + y.*y) / (delta^2);
       Ue = exp(arg);
       Er = Ue - U;
       lmax = max(max(abs(Er)));
       emax = max(emax, lmax);
    end
end

% append results
ktrial = ktrial + 1;
kN(ktrial)  = N;
kdt(ktrial) = dt;
kem(ktrial) = emax;
ktm(ktrial) = nsteps * dt;

% display a short summary (row appended)
nt = length(kN);
it = 1:nt;
disp([it' kN' kdt' kem' ktm' 2*pi ./ kdt']);

end
