function gat = gating_initial(loc, P, typ)
    N = size(loc, 1);
    DG = size(loc, 2);
    switch typ
        case {'constant_sigmoid' 'linear_sigmoid' 'sigmoid'}
            gat = rand(P, DG + 1);
        case {'constant_softmax' 'linear_softmax' 'softmax'}
            gat = rand(P, DG + 1);
        case 'rbf_softmax'
            shu = randperm(N);
            gat = ones(P, DG + 1);
            gat(:, 2:end) = loc(shu(1:P), :);
    end
end
