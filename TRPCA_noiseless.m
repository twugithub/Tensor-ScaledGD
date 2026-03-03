clearvars;
close all;
clc

rng('shuffle');

addpath('tproduct');

n = 100;
n3 = 100;
r = 10;
alpha = 0.1;
kappa_list = [1, 5, 10, 20];

T = 1000;
z0 = 0.5;
z1 = 0.5;
eta = 0.5;
decay = 0.95;
thresh_up = 1e3;
thresh_low = 1e-14;

transform.L = @fft; transform.l = n3; transform.inverseL = @ifft;
% transform.L = @dct; transform.l = 1; transform.inverseL = @idct;

errors_ScaledGD = zeros(length(kappa_list), T);
errors_GD = zeros(length(kappa_list), T);

U_seed = sign(rand(n, r, n3) - 0.5);
[U_star, ~, ~] = tsvd(U_seed, transform, r);
V_seed = sign(rand(n, r, n3) - 0.5);
[V_star, ~, ~] = tsvd(V_seed, transform, r);

for i_kappa = 1:length(kappa_list)

    kappa = kappa_list(i_kappa);
    sigma_star = linspace(1, 1/kappa, r);

    L_star = zeros(n, r, n3);
    R_star = zeros(n, r, n3);
    X_star = zeros(n, n, n3);
    for i = 1:n3
        L_star(:,:,i) = U_star(:,:,i)*diag(sqrt(sigma_star));
        R_star(:,:,i) = V_star(:,:,i)*diag(sqrt(sigma_star));
        X_star(:,:,i) = L_star(:,:,i)*R_star(:,:,i)';
    end

    if isequal(transform.L,@fft)
        X_star = ifft(X_star,[],3);
    else
        X_star = inverselineartransform(X_star,transform);
    end

    S_supp_idx = randsample(n*n*n3, round(alpha*n*n*n3), false);
    S_range = mean(abs(X_star(:)));
    S_temp = 2*S_range*rand(n, n, n3) - S_range;
    S_star = zeros(n, n, n3);
    S_star(S_supp_idx) = S_temp(S_supp_idx);

    Y = X_star + S_star;

    L = zeros(n, r, n3);
    R = zeros(n, r, n3);
    L_plus = zeros(n, r, n3);
    R_plus = zeros(n, r, n3);
    Xtrans = zeros(n, n, n3);

    %% Spectral initialization

    S = thre(Y, z0);
    [U0, Sigma0, V0] = tsvd(Y - S, transform, r);

    if isequal(transform.L,@fft)
        Ytrans = fft(Y,[],3);
    else
        Ytrans = lineartransform(Y,transform);
    end

    %% Scaled GD

    if isequal(transform.L,@fft)

        halfn3 = ceil((n3+1)/2);
        for i = 1 : halfn3
            L(:,:,i) = U0(:,:,i)*sqrt(Sigma0(:,:,i));
            R(:,:,i) = V0(:,:,i)*sqrt(Sigma0(:,:,i));
        end
        for i = halfn3+1 : n3
            L(:,:,i) = conj(L(:,:,n3+2-i));
            R(:,:,i) = conj(R(:,:,n3+2-i));
        end

    else
        for i = 1:n3
            L(:,:,i) = U0(:,:,i)*sqrt(Sigma0(:,:,i));
            R(:,:,i) = V0(:,:,i)*sqrt(Sigma0(:,:,i));
        end
    end

    for t = 1:T
        if isequal(transform.L,@fft)

            halfn3 = ceil((n3+1)/2);
            for i = 1 : halfn3
                Xtrans(:,:,i) = L(:,:,i)*R(:,:,i)';
            end
            for i = halfn3+1 : n3
                Xtrans(:,:,i) = conj(Xtrans(:,:,n3+2-i));
            end

        else
            for i = 1:n3
                Xtrans(:,:,i) = L(:,:,i)*R(:,:,i)';
            end
        end
        if isequal(transform.L,@fft)
            X = ifft(Xtrans,[],3);
        else
            X = inverselineartransform(Xtrans,transform);
        end

        error = norm(X - X_star, 'fro')/norm(X_star, 'fro');
        errors_ScaledGD(i_kappa, t) = error;
        if ~isfinite(error) || error > thresh_up || error < thresh_low
            break;
        end

        if isequal(transform.L,@fft)
            S = thre(Y - X, z1 * (decay^t));
            Strans = fft(S,[],3);
        else
            S = thre(Y - X, z1 * (decay^t));
            Strans = lineartransform(S,transform);
        end

        if isequal(transform.L,@fft)

            halfn3 = ceil((n3+1)/2);
            for i = 1 : halfn3
                L_plus(:,:,i) = L(:,:,i) - eta*(Xtrans(:,:,i) + Strans(:,:,i) - Ytrans(:,:,i))*R(:,:,i)/(R(:,:,i)'*R(:,:,i));
                R_plus(:,:,i) = R(:,:,i) - eta*(Xtrans(:,:,i) + Strans(:,:,i) - Ytrans(:,:,i))'*L(:,:,i)/(L(:,:,i)'*L(:,:,i));
            end
            for i = halfn3+1 : n3
                L_plus(:,:,i) = conj(L_plus(:,:,n3+2-i));
                R_plus(:,:,i) = conj(R_plus(:,:,n3+2-i));
            end

        else
            for i = 1:n3
                L_plus(:,:,i) = L(:,:,i) - eta*(Xtrans(:,:,i) + Strans(:,:,i) - Ytrans(:,:,i))*R(:,:,i)/(R(:,:,i)'*R(:,:,i));
                R_plus(:,:,i) = R(:,:,i) - eta*(Xtrans(:,:,i) + Strans(:,:,i) - Ytrans(:,:,i))'*L(:,:,i)/(L(:,:,i)'*L(:,:,i));
            end
        end
        L = L_plus;
        R = R_plus;
    end

    %% GD

    if isequal(transform.L,@fft)

        halfn3 = ceil((n3+1)/2);
        for i = 1 : halfn3
            L(:,:,i) = U0(:,:,i)*sqrt(Sigma0(:,:,i));
            R(:,:,i) = V0(:,:,i)*sqrt(Sigma0(:,:,i));
        end
        for i = halfn3+1 : n3
            L(:,:,i) = conj(L(:,:,n3+2-i));
            R(:,:,i) = conj(R(:,:,n3+2-i));
        end

    else
        for i = 1:n3
            L(:,:,i) = U0(:,:,i)*sqrt(Sigma0(:,:,i));
            R(:,:,i) = V0(:,:,i)*sqrt(Sigma0(:,:,i));
        end
    end

    for t = 1:T
        if isequal(transform.L,@fft)

            halfn3 = ceil((n3+1)/2);
            for i = 1 : halfn3
                Xtrans(:,:,i) = L(:,:,i)*R(:,:,i)';
            end
            for i = halfn3+1 : n3
                Xtrans(:,:,i) = conj(Xtrans(:,:,n3+2-i));
            end

        else
            for i = 1:n3
                Xtrans(:,:,i) = L(:,:,i)*R(:,:,i)';
            end
        end
        if isequal(transform.L,@fft)
            X = ifft(Xtrans,[],3);
        else
            X = inverselineartransform(Xtrans,transform);
        end

        error = norm(X - X_star, 'fro')/norm(X_star, 'fro');
        errors_GD(i_kappa, t) = error;
        if ~isfinite(error) || error > thresh_up || error < thresh_low
            break;
        end

        if isequal(transform.L,@fft)
            S = thre(Y - X, z1 * (decay^t));
            Strans = fft(S,[],3);
        else
            S = thre(Y - X, z1 * (decay^t));
            Strans = lineartransform(S,transform);
        end

        if isequal(transform.L,@fft)

            halfn3 = ceil((n3+1)/2);
            for i = 1 : halfn3
                L_plus(:,:,i) = L(:,:,i) - eta/sigma_star(1)*(Xtrans(:,:,i) + Strans(:,:,i) - Ytrans(:,:,i))*R(:,:,i);
                R_plus(:,:,i) = R(:,:,i) - eta/sigma_star(1)*(Xtrans(:,:,i) + Strans(:,:,i) - Ytrans(:,:,i))'*L(:,:,i);
            end
            for i = halfn3+1 : n3
                L_plus(:,:,i) = conj(L_plus(:,:,n3+2-i));
                R_plus(:,:,i) = conj(R_plus(:,:,n3+2-i));
            end

        else
            for i = 1:n3
                L_plus(:,:,i) = L(:,:,i) - eta/sigma_star(1)*(Xtrans(:,:,i) + Strans(:,:,i) - Ytrans(:,:,i))*R(:,:,i);
                R_plus(:,:,i) = R(:,:,i) - eta/sigma_star(1)*(Xtrans(:,:,i) + Strans(:,:,i) - Ytrans(:,:,i))'*L(:,:,i);
            end
        end
        L = L_plus;
        R = R_plus;
    end
end



h1 = figure(1);
clrs = {[.5,0,.5], [1,.5,0], [1,0,0], [0,.5,0], [0,0,1]};
mks = {'o', 'x', 'p', 's', 'd'};
set(h1,'position',[100 100 800 600]);
lgd = {};
for i_kappa = 1:length(kappa_list)
    kappa = kappa_list(i_kappa);
    errors = errors_ScaledGD(i_kappa, :);
    errors = errors(errors > thresh_low);
    t_subs = (2*i_kappa):10:length(errors);
    semilogy(t_subs-1, errors(t_subs), 'Color', clrs{1}, 'Marker', mks{i_kappa}, 'MarkerSize', 9);
    hold on; grid on;
    lgd{end+1} = sprintf('$\\mathrm{ScaledGD}~\\kappa=%d$', kappa);
end
for i_kappa = 1:length(kappa_list)
    kappa = kappa_list(i_kappa);
    errors = errors_GD(i_kappa, :);
    errors = errors(errors > thresh_low);
    t_subs = 10:10:length(errors);
    semilogy(t_subs-1, errors(t_subs), 'Color', clrs{2}, 'Marker', mks{i_kappa}, 'MarkerSize', 9);
    hold on; grid on;
    lgd{end+1} = sprintf('$\\mathrm{VanillaGD}~\\kappa=%d$', kappa);
end
xlabel('Iteration count', 'FontSize', 20);
ylabel('Relative error', 'FontSize', 20);
llgd = legend(lgd);
set(llgd, 'Location', 'northeast', 'Interpreter', 'latex', 'FontSize', 20);
legendPos = llgd.Position;
llgd.Position = [legendPos(1)-0.05, legendPos(2), 1.08*legendPos(3), legendPos(4)];



function S = thre(S, theta)
S = sign(S) .* max(abs(S)-theta, 0.0);
end