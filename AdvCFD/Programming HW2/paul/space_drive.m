
jtrial = 0;

for N=20:10:80;
   space_chk
end;

figure; 
semilogy(jN, jem, 'r.', 'MarkerSize', 20);
hold on;
semilogy(jN, jem, 'k--');
hold off;
axis square
title('Spatial Convergence: $t_{\mbox{final}}=2\pi, \; \Delta t=\pi/8000$',intp,ltx,fs,14);
xlabel('$N$',intp,ltx,fs,20); 
ylabel('$\| {\tilde u} - u \|_{\infty}^{}$',intp,ltx,fs,20)

figure;
mesh(X,Y,Er);
drawnow;
