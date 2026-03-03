function [U, S, V] = tsvd(A, transform, r)

[n1,n2,n3] = size(A);

if isequal(transform.L,@fft)
    % efficient computing for fft transform
    A = fft(A,[],3);

    U = zeros(n1,r,n3);
    S = zeros(r,r,n3);
    V = zeros(n2,r,n3);

    halfn3 = ceil((n3+1)/2);
    for i = 1 : halfn3
        [tempU, tempS, tempV] = svd(A(:,:,i));
        U(:,:,i) = tempU(:,1:r);
        S(:,:,i) = tempS(1:r,1:r);
        V(:,:,i) = tempV(:,1:r);
    end
    for i = halfn3+1 : n3
        U(:,:,i) = conj(U(:,:,n3+2-i));
        V(:,:,i) = conj(V(:,:,n3+2-i));
        S(:,:,i) = S(:,:,n3+2-i);
    end
else
    A = lineartransform(A,transform);
    U = zeros(n1,r,n3);
    S = zeros(r,r,n3);
    V = zeros(n2,r,n3);
    for i = 1 : n3
        [tempU, tempS, tempV] = svd(A(:,:,i),'econ');
        U(:,:,i) = tempU(:,1:r);
        S(:,:,i) = tempS(1:r,1:r);
        V(:,:,i) = tempV(:,1:r);
    end
end
end