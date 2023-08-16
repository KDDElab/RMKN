%%  调用说明    20181215  version1@lotus
%   aupr =pr_curve（预测分数,原始标准答案,colour)
%   返回PR曲线和PR曲线下面积aupr
%   PR曲线：precison-recall curve。

function aupr =pr_curve(output,original)
    
    %% 检查是否已将矩阵变成一列
    output=output(:);
    original=original(:);

    %% 按预测结果分数deci降序排序，标准答案roc
    [threshold,ind] = sort(output,'descend');  %[阈值，下标]，把预测分数降序排序
    roc_y = original(ind);    %与阈值预测结果对应的标准答案

    %% 求x轴recall的各个点，求y轴precison的各个点。求PR曲线下面积aupr
    P=[1:length(roc_y)]';   %实际上是求(TP+FP)，即所有预测为阳的个数。因为阈值已是降序排序，阈值对应的下标即（TP+FP）
    stack_x = cumsum(roc_y == 1)/sum(roc_y == 1); %x轴：TPR=recall=TP/(TP+FN)=预测为阳的正类/所有正类
    stack_y = cumsum(roc_y == 1)./P; %y轴：precision=TP/(TP+FP)=预测为阳的正类/所有预测为阳
    aupr=sum((stack_x(2:length(roc_y))-stack_x(1:length(roc_y)-1)).*stack_y(2:length(roc_y)));  %PR曲线下面积

    %% 画PR曲线
%     % subplot(2,2,1);   %把绘图窗口分成两行两列四块区域，然后在每个区域分别作图。在第一块绘图
%     %figure;
%     plot(stack_x,stack_y,colour);
%     xlabel('recall');
%     ylabel('precision');

end

