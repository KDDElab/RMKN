clear all
clc
res_all = [];
for data_num = 1
    get_dnames; 
    dnames = [dname];
    tot_data = load([dnames '.mat']);
    data_only  = tot_data.X(:,1:end-1);
    data_labels = tot_data.y(:,end);
    %data_only = tot_data.X; % 所有样本（只有特征）
    %data_labels = tot_data.y; %所有样本的标签
    data_label = data_labels;
    count = 0;
    
    %%%%%%%%%%%%% 数据预处理 %%%%%%%%%%%%%%
    res.dname=[dname num2str(1)];
    pos_class = 0;
    data_label(data_labels==pos_class) = 1;
    data_label(data_labels~=pos_class) = 2;
    target_data = data_only(data_label==1,:);
    target_data = cat(2, target_data, ones(size(target_data,1),1));
    outlier_data = data_only(data_label==2,:);
    outlier_data = cat(2, outlier_data, ones(size(outlier_data,1),1).*2);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    tot_fold =5;
    [m1,n1] = size(target_data);
    ind_pos_all = crossvalind('kfold', target_data(1:m1,n1),5);
    [m2,n2] = size(outlier_data);
    ind_neg_all = crossvalind('kfold', outlier_data(1:m2,n2),5);
    %     %load([dname '5pos']);
    %     load([dname '5' 'pos']);
    %     ind_pos_all = fold_run_pos;
    %     %load([dname '5neg']);
    %     load([dname '5' 'neg']);
    %     ind_neg_all = fold_run_neg;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    optmodf = {};
    kk = 1;
    dis_knn = 3; %图的参数
    %     JNN = 2; % 测试点到最近邻训练点的个数
    %     KNN = 2; % 最近邻训练点到其对应的最近邻的个数
    tic
    typical = 1;
    for JNN = 2:6
        for KNN = 2:6
            for run = 1
                ind_pos = ind_pos_all(:,run);
                ind_neg = ind_neg_all(:,run);
                for f = 1:tot_fold
                    test_posind = (ind_pos == f);
                    train_posind =~ test_posind;
                    test_negind = (ind_neg == f);
                    train_negind =~ test_negind;
                    
                    test_pos = target_data(test_posind,:);
                    train_pos = target_data(train_posind,:);
                    test_neg = outlier_data(test_negind,:);
                    train_neg = outlier_data(train_negind,:);
                    
                    %%%%%%%%%%%%% train,test,val 数据集 %%%%%%%%%%%%%%
                    
                    train_data = train_pos(:,1:end-1);
                    train_lbls = train_pos(:,end);
                    test_data = cat(1,test_pos(:,1:end-1),test_neg(:,1:end-1));
                    test_lbls = cat(1,test_pos(:,end),test_neg(:,end));
                    val_data = train_neg(:,1:end-1);
                    val_lbls = train_neg(:,end);
                    
                    train_number = sum(train_lbls==2);
                    test_number = sum(test_lbls==2);
                    val_number = sum(val_lbls==2);
                    
                    mmod.nor.dat = mean_and_std(train_data,'true');
                    train_data_m = normalize_data(train_data,mmod.nor.dat);%训练集归一化
                    mmod.nor.dat = mean_and_std(test_data,'true');
                    test_data_m = normalize_data(test_data,mmod.nor.dat);
                    mmod.nor.dat = mean_and_std(val_data,'true');
                    val_data_m = normalize_data(val_data,mmod.nor.dat);
                    
                    %%%%%%%%%%%%%%%% graph_weights %%%%%%%%%%%%%%%%
                    
                    [num_data,dim] = size(train_data_m);
                    distance = zeros(num_data,num_data);
                    for i = 1:num_data
                        for j = 1:num_data
                            distance(i,j) = sqrt(sum((train_data_m(i,:)-train_data_m(j,:)).^2));
                        end
                    end
                    [dis_value,dis_index] = sort(distance);
                    
                    knn_index = dis_index(2:dis_knn+1,:);
                    
                    graph_weights = ones(num_data,num_data); %这里看看要不要改一改
                    for i = 1:num_data
                        for j = 1:dis_knn
                            graph_weights(i,knn_index(j,i)) = (train_data_m(i,:)*train_data_m(knn_index(j,i),:)'+1)^2;%使用多项式核函数
                        end
                    end
                    
                    %%%%%%%%%%%%%%%% begin training %%%%%%%%%%%%%%%%%
                    temp_ind = 0;
                    optmod = {};
                    
                    num_kern = 3;
                    sig_array = power(2,-6:6);
                    for sigm = 1:length(sig_array)
                        for k = 2:3
                            %for c = 0.1:0.1:0.9
                            final = {['g' num2str(sig_array(sigm))],['p' num2str(k)],'l'};
                            count = count + 1;
                            [data_num JNN-1 KNN-1 count]
                            [model,eta] = create_model(train_data_m,train_lbls,final,graph_weights);
                            
                            [labels_GOCK,labels_score] = test_model(test_data_m,model,final,JNN,KNN,test_number);
                            [labeltr_GOCK,labeltr_score] = test_model(train_data_m,model,final,JNN,KNN,train_number);
                            [labelval_GOCK,labelval_score] = test_model(val_data_m,model,final,JNN,KNN,val_number);
                            
                            act_val_lbls = cat(1, train_pos(:,end), train_neg(:,end));
                            pred_val_lbls = [labeltr_GOCK; labelval_GOCK];
                            pred_val_score = [labeltr_score; labelval_score];
                            temp_ind = temp_ind +1;
                            optmod{temp_ind} = model;
                            
                            [accu(temp_ind) sens(temp_ind) spec(temp_ind) prec(temp_ind) rec(temp_ind) f11(temp_ind) gm(temp_ind) auprtr(temp_ind)] = Evaluate(act_val_lbls,pred_val_lbls,pred_val_score,1);
                            [accut(temp_ind) senst(temp_ind) spect(temp_ind) prect(temp_ind) rect(temp_ind) f11t(temp_ind) gmt(temp_ind) auprt(temp_ind)] = Evaluate(test_lbls,labels_GOCK,labels_score,1);
                            
                            clear labels_GOCK labelval_GOCK labeltr_GOCK;
                            
                        end
                        kk = kk+1;
                    end
                    %end
                    %%%% Choose optimal parameters based on performance on validatin set
                    [max_val opt_ind] = max(auprtr);
                    [max_valt opt_indt] = max(auprt); %%% there is no use of opt_indt. just used for checking someting
                    %sup_vecf=[sup_vecf;sup_vec(opt_ind)];
                    res.optmodf{run,f} = optmod{opt_ind};
                    %%%% Training and Validation
                    traccuracy(f)=accu(opt_ind);
                    trsensitivity(f)=sens(opt_ind);
                    trspecificity(f)=spec(opt_ind);
                    trprecision(f)=prec(opt_ind);
                    trrecall(f)=rec(opt_ind);
                    trf1(f)=f11(opt_ind);
                    trgmean(f) = gm(opt_ind);
                    traupr(f) = auprtr(opt_ind);
                    
                    %%%% Testing
                    accuracy(f)=accut(opt_ind); sensitivity(f)=senst(opt_ind); specificity(f)=spect(opt_ind);
                    precision(f)=prect(opt_ind); recall(f)=rect(opt_ind); f1(f)=f11t(opt_ind); gmean(f)=gmt(opt_ind);
                    aupr(f) = auprt(opt_ind);
                    clear test_posind train_posind test_negind train_negind test_pos...
                        train_pos test_neg train_neg train_data train_lbls test_data...
                        test_lbls Ktrain Ktest labels_GOCK gm
                end
                
                %%% For Training
                mtraccuracy(run) = mean(traccuracy)*100; mtrsensitivity(run) = mean(trsensitivity)*100; mtrspecificity(run) = mean(trspecificity)*100;
                mtrprecision(run) = mean(trprecision)*100; mtrrecall(run) = mean(trrecall)*100; mtrf1(run) = mean(trf1)*100; mtrgmean(run) = mean(trgmean)*100;
                mtraupr(run) = mean(traupr)*100;
                train_result = [trsensitivity'*100 trspecificity'*100 trprecision'*100 trrecall'*100 trf1'*100 traccuracy'*100 trgmean'*100 traupr'*100];
                mtrain_result = [mtrsensitivity(run) mtrspecificity(run) mtrprecision(run) mtrrecall(run) mtrf1(run) mtraccuracy(run) mtrgmean(run) mtraupr(run)];
                %msv(run) = mean(sup_vecf);
                %res.svfold(:,run) = sup_vecf';
                %        res.svmeanfold(run) = msv(run);
                res.trainfold(:,:,run) = train_result;
                res.trainmeanfold(run,:) = mtrain_result;
                clear traccuracy trsensitivity trspecificity trprecision trrecall trf1 trgmean train_result mtrain_result...
                    ind_pos ind_neg;
                
                %%% For Testing
                maccuracy(run) = mean(accuracy)*100; msensitivity(run) = mean(sensitivity)*100; mspecificity(run) = mean(specificity)*100;
                mprecision(run) = mean(precision)*100; mrecall(run) = mean(recall)*100; mf1(run) = mean(f1)*100; mgmean(run) = mean(gmean)*100;
                maupr(run) = mean(aupr)*100;
                test_result = [sensitivity'*100 specificity'*100 precision'*100 recall'*100 f1'*100 accuracy'*100 gmean'*100 aupr'*100];
                mtest_result = [msensitivity(run) mspecificity(run) mprecision(run) mrecall(run) mf1(run) maccuracy(run) mgmean(run) maupr(run)];
                res.testfold(:,:,run) = test_result;
                res.testmeanfold(run,:) = mtest_result;
                clear ind_pos ind_neg accuracy sensitivity specificity precision recall f1 gmean auc aupr test_result mtest_result;
            end
            toc
            %%% For Training
            %%%% Final performance evaluation over 5 runs
            mmtraccuracy = mean(mtraccuracy); mmtrsensitivity = mean(mtrsensitivity); mmtrspecificity = mean(mtrspecificity);
            mmtrprecision = mean(mtrprecision); mmtrrecall = mean(mtrrecall); mmtrf1 = mean(mtrf1); mmtrgmean = mean(mtrgmean);
            mmtraupr = mean(mtraupr);
            %%% For Testing
            %%%% Final performance evaluation over 5 runs
            mmaccuracy = mean(maccuracy); mmsensitivity = mean(msensitivity); mmspecificity = mean(mspecificity);
            mmprecision = mean(mprecision); mmrecall = mean(mrecall); mmf1 = mean(mf1); mmgmean = mean(mgmean);
            mmaupr = mean(maupr);
            %%% For Training
            %%%% Standard Deviation over 5 runs
            std_mmtraccuracy = std(mtraccuracy); std_mmtrsensitivity = std(mtrsensitivity); std_mmtrspecificity = std(mtrspecificity);
            std_mmtrprecision = std(mtrprecision); std_mmtrrecall = std(mtrrecall); std_mmtrf1 = std(mtrf1); std_mmtrgmean = std(mtrgmean);
            std_mmtraupr = std(mtraupr);
            %%% For Testing
            %%%% Standard Deviation over 5 runs
            std_mmaccuracy = std(maccuracy); std_mmsensitivity = std(msensitivity); std_mmspecificity = std(mspecificity);
            std_mmprecision = std(mprecision); std_mmrecall = std(mrecall); std_mmf1 = std(mf1); std_mmgmean = std(mgmean);
            std_mmaupr = std(maupr);
            
            mtrain_result = [mtrsensitivity' mtrspecificity' mtrprecision' mtrrecall' mtrf1' mtraccuracy' mtrgmean' mtraupr' toc];
            mmtrain_result = [mmtrsensitivity mmtrspecificity mmtrprecision mmtrrecall mmtrf1 mmtraccuracy mmtrgmean mmtraupr toc];
            res.trainmeanrun = mmtrain_result;
            %res.svmeanrun=mean(msv);
            J = JNN-1; K = KNN-1;
            mtest_result = [msensitivity' mspecificity' mprecision' mrecall' mf1' maccuracy' mgmean' maupr' toc];
            mmtest_result = [mmsensitivity mmspecificity mmprecision mmrecall mmf1 mmaccuracy mmgmean mmaupr toc J K];
            store_acc(1)=mmaccuracy;
            store_gmean(1)=mmgmean;
            res.testmeanrun = mmtest_result;
            
            mmtrainstd_result = [std_mmtrsensitivity std_mmtrspecificity std_mmtrprecision std_mmtrrecall std_mmtrf1 std_mmtraccuracy std_mmtrgmean std_mmtraupr toc J K];
            res.trainstd = mmtrainstd_result;
            %res.svstd=std(msv);
            mmteststd_result = [std_mmsensitivity std_mmspecificity std_mmprecision std_mmrecall std_mmf1 std_mmaccuracy std_mmgmean std_mmaupr toc J K];
            res.teststd = mmteststd_result;
            
            clear   mmtrain_result mmtrainstd_result mtrain_result mtest_result;
            res.performance = {'sensitivity','specificity', 'precision', 'recall', 'f1', 'accuracy', 'gmean', 'aupr','time','KNN','JNN'};
            res_all{data_num,typical} = res;
            clear res;
            save(sprintf('%s%d%s','res',data_num,'.mat'),'res_all')
            typical = typical +1;
        end
    end
end
save('gpl_result.mat','res_all')