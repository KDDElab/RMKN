   function gra = eta_gradient_lmk(Km, eta, loc, gat, typ,graph_w)
    N = size(loc, 1);
    DG = size(loc, 2) - 1;
    P = size(Km, 3);
    gra = zeros(P, DG + 1);
    switch typ
        case {'linear_sigmoid', 'sigmoid'}
            for m = 1:P
                first = eta(:, m) * eta(:, m)'.* Km(:,:,m).*graph_w;
                second = eta(:, m) * eta(:, m)'.* Km(:,:,m).*graph_w;
                for f = 1:DG + 1
                    temp = repmat((1 - eta(:, m)) .* loc(:, f), 1, N);
                    gra(m, f) = gra(m, f) -2 * sum(sum(first .* (temp + temp'))) + sum(sum(diag(second .*(temp + temp)))) + sum(sum(diag(second .*(temp' + temp'))));%改了
                end
            end
        case {'linear_softmax', 'softmax'}
            for h = 1:P
                first = ((alp .* eta(:, h)) * (alp .* eta(:, h))') .* Km(:, :, h);
                second = ((alp.* eta(:,h)) .* eta(:,h)).* diag(Km(:,:,h));
                for m = 1:P
                    del = (m == h);
                    for f = 1:DG + 1
                        temp = repmat((del - eta(:, m)) .* loc(:, f), 1, N);
                        gra(m, f) = gra(m, f) - 0.5 * sum(sum(first .* (temp + temp')))+ sum(sum(second .* diag(temp + temp)));
                    end
                end
            end
        case {'rbf_softmax', 'rbf'}
            for h = 1:P
                first = ((alp .* eta(:, h)) * (alp .* eta(:, h))') .* Km(:, :, h);
                second = ((alp.* eta(:,h)) * eta(:,h)').* diag(Km(:,:,h));
                for m = 1:P
                    del = (m == h);
                    temp = repmat((del - eta(:, m)) .* sum(bsxfun(@minus, loc(:, 2:end), gat(m, 2:end)).^2, 2) / gat(m, 1)^3, 1, N);
                    gra(m, 1) = gra(m, 1) - sum(sum(first .* (temp + temp')))+ sum(sum(second .* diag(temp + temp)));
                    for f = 2:DG + 1
                        temp = repmat((del - eta(:, m)) .* (loc(:, f) - gat(m, f)) / gat(m, 1)^2, 1, N);
                        gra(m, f) = gra(m, f) - sum(sum(first .* (temp + temp')))+ sum(sum(second .* diag(temp + temp)));
                    end
                end
            end
                case {'constant_sigmoid'}
            for m = 1:P
                first = ((alp .* eta(:, m)) * (alp .* eta(:, m))') .* Km(:, :, m);
                temp = repmat((1 - eta(:, m)) .* loc(:, 1), 1, N);
                gra(m, 1) = gra(m, 1) - 0.5 * sum(sum(first .* (temp + temp')));
            end
        case {'constant_softmax'}
            for h = 1:P
                first = ((alp .* eta(:, h)) * (alp .* eta(:, h))') .* Km(:, :, h);
                for m = 1:P
                    del = (m == h);
                    temp = repmat((del - eta(:, m)) .* loc(:, 1), 1, N);
                    gra(m, 1) = gra(m, 1) - 0.5 * sum(sum(first .* (temp + temp')));
                end
            end  
    end    
end
