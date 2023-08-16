function ker = get_kernel(nam)
    sub = strfind(nam, '@');
    if isempty(sub) == 1
        switch nam(1)
            case 'l'
                ker = @(x, y)(x * y');
            case 'p'
                q = str2num(nam(2:end)); %#ok<ST2NM>
                ker = @(x, y)(x * y' + 1) .^ q;
            case 'g'
                s = str2num(nam(2:end)); %#ok<ST2NM>
                ker = @(x, y)exp((2 * x * y' - repmat(sqrt(sum(x .^ 2, 2) .^ 2), 1, size(y, 1)) - repmat(sqrt(sum(y .^ 2, 2)' .^ 2), size(x, 1), 1)) / s^2);
            case 'a'
                t = str2num(nam(2:end)); %#ok<ST2NM>
                ker = @(x, y)exp(-t*pdist2(x,y,'cityblock'));
            case 's'
                m = str2num(nam(2:end)); %#ok<ST2NM>
                ker = @(x, y)tanh(m*x*y');
        end
    else
        ind = str2num(nam(sub + 1:end)); %#ok<ST2NM>
        switch nam(1)
            case 'l'
                ker = @(x, y)(x(:, ind) * y(:, ind)');
            case 'p'
                q = str2num(nam(2:sub - 1)); %#ok<ST2NM>
                ker = @(x, y)(x(:, ind) * y(:, ind)' + 1) .^ q;
            case 'g'
                s = str2num(nam(2:sub - 1)); %#ok<ST2NM>
                ker = @(x, y)exp((2 * x(:, ind) * y(:, ind)' - repmat(sqrt(sum(x(:, ind) .^ 2, 2) .^ 2), 1, size(y, 1)) - repmat(sqrt(sum(y(:, ind) .^ 2, 2)' .^ 2), size(x, 1), 1)) / s^2);
            case 'a'
                t = str2num(nam(2:sub - 1)); %#ok<ST2NM>
                ker = @(x, y)exp(-t*pdist2(x(:,ind),y(:,ind),'cityblock'));
            case 's'
                m = str2num(nam(2:sub - 1)); %#ok<ST2NM>
                ker = @(x, y)tanh(m*x(:,ind)*y(:,ind)');
        end
    end
end
