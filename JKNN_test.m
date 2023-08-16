function [out] = JKNN_test(tes, mod,JNN,KNN,number)
[N,~] = size(tes{1}.X);
[num_data,~]=size(mod.sup{1}.X);
P = length(tes) - 1;
for m = 1:P
    tes{m}.X = normalize_data(tes{m}.X, mod.nor.dat{m});
end
loc = locality(tes{P + 1}.X, mod.par.loc.typ);
loc = normalize_data(loc, mod.nor.loc);
out.eta = etas(loc, mod.gat, mod.par.eps, mod.par.gat.typ);

K_tetr = zeros(N,num_data);
K_tete = zeros(N,N);
%dis = zeros(N,N);
for m = 1:P
    K_tetr = K_tetr + out.eta(:,m).*kernel(tes{m},mod.sup{m},mod.par.ker{m},mod.par.nor.ker{m});
    K_tete = K_tete + out.eta(:,m).*out.eta(:,m).*kernel(tes{m}, tes{m}, mod.par.ker{m}, mod.par.ker{m});
end
   
dis_JNN = zeros(num_data,N);
for i = 1:N
    for j = 1:num_data
        dis_JNN(j,i) = K_tete(i,i)-2*K_tetr(i,j)+mod.yyKeta(j,j);
    end
end

[disJNN_value,JNN_index] = sort(dis_JNN);
ave_disJNN = sum(disJNN_value(1:JNN,:))/JNN;


JNN_index = JNN_index(1:KNN,:);
dis_KNN = zeros(num_data,num_data);
for i = 1:num_data
    for j = 1:num_data
        dis_KNN(i,j) = mod.yyKeta(i,i)-2*mod.yyKeta(i,j)+mod.yyKeta(j,j);
    end
end

[disKNN_value,~] = sort(dis_KNN);
disKNN_valueKNN = sum(disKNN_value(1:KNN,:));

ave_disKNN = zeros(1,N);
for i = 1:N
    temp = 0;
    for j = 1:KNN %%%%
        temp = temp + disKNN_valueKNN(1,JNN_index(j,i));
    end
    ave_disKNN(1,i) = temp/(KNN*JNN);
end

AA = ave_disJNN./ave_disKNN;
out.outlier_score = mapminmax(AA,0,1)';
aa_value = sort(AA);
if number==0
    out.predictedlabel = ones(N,1);
else
    %theta = aa_value(1,length(aa_value)-fix(number));
    theta = aa_value(1,length(aa_value)-number+1);
    out.predictedlabel = ones(N,1);
    out.predictedlabel(AA >= theta) = 2;
end
