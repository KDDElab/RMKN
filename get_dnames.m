clearvars -except a b res_all data_num path res_acc res_gmean;

if data_num == 1
    dname = 'wine';
    dnames = [path dname];
end

if data_num == 2
    dname = 'glass';
    dnames = [path dname];
end

if data_num == 3
    dname = 'vertebral';
    dnames = [path dname];
end

if data_num == 5
    dname = 'wbc';
    dnames = [path dname];
end

if data_num == 4
    dname = 'ionosphere';
    dnames = [path dname];
end

if data_num == 6
    dname = 'arrhythmia';
    dnames = [path dname];
end

if data_num == 7
    dname = 'breastw';
    dnames = [path dname];
end

if data_num == 8
    dname = 'pima';
    dnames = [path dname];
end

if data_num == 9
    dname = 'vowels';
    dnames = [path dname];
end

if data_num == 10
    dname = 'musk';
    dnames = [path dname];
end

if data_num == 11
    dname = 'speech';
    dnames = [path dname];
end

if data_num == 12
    dname = 'thyroid';
    dnames = [path dname];
end

if data_num == 13
    dname = 'satimage-2';
    dnames = [path dname];
end

if data_num == 13
    dname = 'letter';
    dnames = [path dname];
end

if data_num == 14
    dname = 'pendigits';
    dnames = [path dname];
end

if data_num == 15
    dname = 'satellite';
    dnames = [path dname];
end

if data_num == 16
    dname = 'annthyroid';
    dnames = [path dname];
end

if data_num == 17
    dname = 'mnist';
    dnames = [path dname];
end

if data_num == 18
    dname = 'mammography';
    dnames = [path dname];
end
