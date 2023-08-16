function [mod,eta] = mkljknn_train(tra, par, graph_w)
rand('twister',par.see)
P = length(tra) - 1;
for m = 1:P
    mod.nor.dat{m} = mean_and_std(tra{m}.X, par.nor.dat{m});
    tra{m}.X = normalize_data(tra{m}.X, mod.nor.dat{m});
end
mod.loc = locality(tra{P + 1}.X, par.loc.typ);
mod.nor.loc = mean_and_std(mod.loc, par.nor.loc);
mod.loc = normalize_data(mod.loc, mod.nor.loc);
mod.gat = gating_initial(mod.loc, P, par.gat.typ);
eta = etas(mod.loc, mod.gat, par.eps, par.gat.typ); %初始化多核权重
N = size(tra{1}.X, 1);
yyKm = zeros(N, N, P);
for m = 1:P
    yyKm(:, :, m) = (tra{m}.y * tra{m}.y') .* kernel(tra{m}, tra{m}, par.ker{m}, par.nor.ker{m});
end
yyKeta = kernel_eta(yyKm, eta); % 多核权重+核函数的最后组合w(xi)k(xi,xj)w(xj)
mod.yyKold = yyKeta;

%%%%%%%%%%%%%%%%% object function %%%%%%%%%%%%%%%%

obj = objectfunction(tra{1},yyKeta,graph_w);

%%%%%%%%%%%%%%%%% training MKL %%%%%%%%%%%%%%%%%%%%%
mod.obj = obj;
mod.sol = 1;

while 1&&P > 1
    oldObj = obj;
    [eta,mod,obj,yyKeta] = learn_eta(tra,par,yyKm,eta,mod,obj,yyKeta,graph_w);
    mod.obj = [mod.obj, obj];
    if abs(obj - oldObj) <= par.eps * abs(oldObj)
        break;
    end
end
mod.yyKeta = yyKeta;
mod = rmfield(mod, 'loc');
for m = 1:P
    mod.sup{m}.ind = tra{m}.ind;
    mod.sup{m}.X = tra{m}.X;
    mod.sup{m}.y = tra{m}.y;
    mod.sup{m}.eta = eta(m);
end
mod.par = par;



    
    
    
    
    
    
    
    
    
    

end