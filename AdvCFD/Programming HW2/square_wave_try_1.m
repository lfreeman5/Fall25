% SOLVE -d^2 u" + u = f on [0,pi] (Legendre) and [-pi,pi] (Fourier) %
% for f=square-wave.
format compact; format longe; hold off; clear
N=256; M=400; %% Over-sample for error checking 
L = pi; d=1; d2=d*d;
[Ah,Bh,Ch,Dh,z,w]=semhat(M);
Ab = (2/L)*Ah; Bb = (L/2)*Bh; Ib = eye(M+1); R = Ib(2:end-1,:); 
A=R*Ab*R'; B=R*Bb*R'; H = d2*A + B;

z=L*(1+z)/2; w=L*w/2;
xb=z; fb = 1+0*xb;
fb = pi^2/4 - (xb-pi/2).^2;
us= R'*(H\(R*Bb*fb)); 

% %% Exact solution for -u'' + u = 1, u(0)=u(pi)=0
% ue = -cosh(xb) + ((cosh(pi)-1)/sinh(pi))*sinh(xb) + 1;
%% Exact solution for -u'' + u = pi*x - x^2, u(0)=u(pi)=0
ue = ((1 - exp(-pi))/sinh(pi))*exp(xb) + ((exp(pi) - 1)/sinh(pi))*exp(-xb) ...
     - xb.^2 + pi*xb - 2;

%% First plot: Legendre, Fourier, and exact solution
figure(1); clf
plot(xb,us,'k-','linewidth',1.4); hold on;
plot(xb,ue,'m-.','linewidth',1.4);  % exact


uf=0*xb; %% Initialize Fourier solution on GLL pts 
for k=1:2:N; k1=k+1;
    sk = sin(k*xb);
    % fk = 2/(k*pi);
    fk = 8/(pi*k^3);
    uk = fk./(1+d2*k*k);
    uf = uf + uk*sk;
    ef=uf-us;
    em(k1/2)=max(abs(ef));
    ef2(k1/2)= sqrt(ef' *Bb*ef/pi);
    nN(k1/2) = k;
    if k==1;  plot(xb,uf,'r--'); end
    if k==3;  plot(xb,uf,'g--'); end
    if k==5;  plot(xb,uf,'b--'); end
end

title('Legendre and Fourier Spectral Method for -u"+u=\pi x -x^2','fontsize',12)
xlabel('x','fontsize',14); ylabel('u(x)','fontsize',14);
axis square; axis([0 pi 0 .7]);
legend('Legendre','Exact','Fourier k=1','Fourier k=3','Fourier k=5')
print -dpng 'legendre_fourier.png'; hold off

%% Second plot: Convergence of Fourier spectral method
figure(2); clf
loglog(nN,ef2,'b.-',nN,0.3*nN.^(-2),'r.-',nN,em,'k.-','linewidth',1.3)
title('Convergence of Fourier Spectral Method for -u"+u=\pi x -x^2','fontsize',12)
xlabel('N','fontsize',14); ylabel('Error: L^2 and L^\infty','fontsize',14);
axis square; axis([0 N 1e-7 2]); 
legend('L^2 error','N^{-2} decay', 'L^\infty error')
text(9,2e-5,'Slope = -2.5','fontsize',12); 
text(20,1e-3,'Slope = -2','fontsize',12);
print -dpng 'fourier_sq_convergence.png'
