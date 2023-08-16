function [eta,mod,obj,yyKeta] = learn_eta(tra,par,yyKm,eta,mod,obj,yyKeta,graph_w)
[num_data,~]=size(tra{1}.X);
gra = eta_gradient_lmk(yyKm,eta,[ones(num_data, 1), mod.loc],mod.gat, par.gat.typ,graph_w);
srn = sqrt(sum(sum(gra.^2)));
if srn ~= 0
    gra = gra ./ srn;
else
    return
end
coe = 1;
oldCoe = 0;
oldObj = obj;
while 1
    if coe >= 1
        oldEta = eta; oldMod = mod; oldObj = obj;
    end
    mod.gat = mod.gat - (coe - oldCoe) * gra;
    eta = etas(mod.loc, mod.gat, par.eps, par.gat.typ);
    yyKeta = kernel_eta(yyKm, eta);
    obj = objectfunction(tra,yyKm,graph_w);
    mod.sol = mod.sol + 1;
    if obj < oldObj
        if coe >= 1
            oldCoe = coe;
            coe = coe * 2;
        else
            break;
        end
    else
        if coe > 1
            oldMod.sol = mod.sol;
            eta = oldEta; mod = oldMod; obj = oldObj;
            yyKeta = kernel_eta(yyKm, eta);
            break;
        else
            oldCoe = coe;
            coe = coe / 2;
            if (coe < par.eps)
                oldMod.sol = mod.sol;
                eta = oldEta; mod = oldMod; obj = oldObj;
                yyKeta = kernel_eta(yyKm, eta);
                break;
            end
        end
    end
end
end