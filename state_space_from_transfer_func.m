iq = tf([0,1],[1,0],1); % transfer function in z
iq.Variable = 'q^-1'; % change representation to inv q

% System from "Identification of Hammerstein-Wiener models"
% A. Wills, T. Schön, Lennart Ljung, and Brett Ninness
% 2013, Automatica (49),1,70-81
% http://dx.doi.org/10.1016/j.automatica.2012.09.018

GXX_den = 1 - 0.77*iq - 0.56*iq^2 + 0.38*iq^3 + 0.012*iq^4;
G11_num = 1.1 - 0.99*iq - 0.17*iq^2 + 0.51*iq^3 - 0.18*iq^4;
G12_num = 0.35*iq - 0.31*iq^2 - 0.24*iq^3 + 0.066*iq^4;
G21_num = -0.86 + 0.39*iq + 0.40*iq^2 - 0.20*iq^3 + 0.012*iq^4;
G22_num = -0.12*iq + 0.15*iq^2 + 0.12*iq^3 - 0.0033*iq^4;

G11 = G11_num / GXX_den;
G12 = G12_num / GXX_den;
G21 = G21_num / GXX_den;
G22 = G22_num / GXX_den;

G = [G11,G12;G21,G22];

% figure;
% step(G)
% figure;
% step(SS)

% simulate system to obtain data:
x0 = zeros(1,4);
u = randn(2,5000)';
y = lsim(G,u,0:4999,x0);
data = iddata(y,u,1);

% subspace method + pem to get good approximation
opts = n4sidOptions('InitialState',x0,'Focus','simulation');
approx_sys = n4sid(data,4,'Feedthrough',[true,true],'DisturbanceModel','none',opts);
ided_sys = pem(data,approx_sys);

figure;
step(G);
figure;
step(ided_sys);

ided_sys


