clearvars;
close all;
clc

rng('shuffle');

addpath('tproduct');

n = 50;
n3 = 20;
r = 5;
m = 5*n*n3*r;
kappa_list = [1, 5, 10, 20];

T = 1000;
eta = 0.5;
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
As = cell(m, 1);
for k = 1:m
    As{k} = randn(n, n, n3)/sqrt(m);
end

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

    y = zeros(m, 1);
    for k = 1:m
        y(k) = As{k}(:)'*X_star(:);
    end

    L = zeros(n, r, n3);
    R = zeros(n, r, n3);
    L_plus = zeros(n, r, n3);
    R_plus = zeros(n, r, n3);
    Xtrans = zeros(n, n, n3);

    %% Spectral initialization

    Y = zeros(n, n, n3);
    for k = 1:m
        Y = Y + y(k)*As{k};
    end
    [U0, Sigma0, V0] = tsvd(Y, transform, r);

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

        Z = zeros(n, n, n3);
        for k = 1:m
            Z = Z + (As{k}(:)'*X(:) - y(k))*As{k};
        end
        if isequal(transform.L,@fft)
            Ztrans = fft(Z,[],3);
        else
            Ztrans = lineartransform(Z,transform);
        end

        if isequal(transform.L,@fft)

            halfn3 = ceil((n3+1)/2);
            for i = 1 : halfn3
                L_plus(:,:,i) = L(:,:,i) - eta*Ztrans(:,:,i)*R(:,:,i)/(R(:,:,i)'*R(:,:,i));
                R_plus(:,:,i) = R(:,:,i) - eta*Ztrans(:,:,i)'*L(:,:,i)/(L(:,:,i)'*L(:,:,i));
            end
            for i = halfn3+1 : n3
                L_plus(:,:,i) = conj(L_plus(:,:,n3+2-i));
                R_plus(:,:,i) = conj(R_plus(:,:,n3+2-i));
            end

        else
            for i = 1:n3
                L_plus(:,:,i) = L(:,:,i) - eta*Ztrans(:,:,i)*R(:,:,i)/(R(:,:,i)'*R(:,:,i));
                R_plus(:,:,i) = R(:,:,i) - eta*Ztrans(:,:,i)'*L(:,:,i)/(L(:,:,i)'*L(:,:,i));
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

        Z = zeros(n, n, n3);
        for k = 1:m
            Z = Z + (As{k}(:)'*X(:) - y(k))*As{k};
        end
        if isequal(transform.L,@fft)
            Ztrans = fft(Z,[],3);
        else
            Ztrans = lineartransform(Z,transform);
        end

        if isequal(transform.L,@fft)

            halfn3 = ceil((n3+1)/2);
            for i = 1 : halfn3
                L_plus(:,:,i) = L(:,:,i) - eta/sigma_star(1)*Ztrans(:,:,i)*R(:,:,i);
                R_plus(:,:,i) = R(:,:,i) - eta/sigma_star(1)*Ztrans(:,:,i)'*L(:,:,i);
            end
            for i = halfn3+1 : n3
                L_plus(:,:,i) = conj(L_plus(:,:,n3+2-i));
                R_plus(:,:,i) = conj(R_plus(:,:,n3+2-i));
            end

        else
            for i = 1:n3
                L_plus(:,:,i) = L(:,:,i) - eta/sigma_star(1)*Ztrans(:,:,i)*R(:,:,i);
                R_plus(:,:,i) = R(:,:,i) - eta/sigma_star(1)*Ztrans(:,:,i)'*L(:,:,i);
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