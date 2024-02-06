function mop = testmop(testname, dimension, num_objecitves, type)
%Get test multi-objective problems from a given name.
%   The method get testing or benchmark problems for Multi-Objective
%   Optimization. The implemented problems included ZDT, OKA, KNO.
%   User get the corresponding test problem, which is an instance of class
%   mop, by passing the problem name and optional dimension parameters.
    mop = struct('name', [], 'od', [], 'pd', [],  'type', [],'domain', [], 'func', []);
    switch lower(testname)
        case 'kno1'
            mop = kno1(mop);
        case 'zdt1'
            mop = zdt1(mop, dimension);
        case 'zdt2'
            mop = zdt2(mop, dimension);
        case 'zdt3'
            mop = zdt3(mop, dimension);
        case 'zdt4'
            mop = zdt4(mop, dimension);
        case 'zdt6'
            mop = zdt6(mop, dimension);
        case 'dtlz1'
            mop = dtlz1(mop, dimension, num_objecitves);
        case 'dtlz2'
            mop = dtlz2(mop, dimension, num_objecitves);
        case 'dtlz3'
            mop = dtlz3(mop, dimension, num_objecitves);
        case 'dtlz4'
            mop = dtlz4(mop, dimension, num_objecitves);
        case 'dtlz5'
            mop = dtlz5(mop, dimension, num_objecitves);
        case 'dtlz5i'
            mop = dtlz5i(mop, dimension, num_objecitves);
        case 'dtlz6'
            mop = dtlz6(mop, dimension, num_objecitves);
        case 'dtlz7'
            mop = dtlz7(mop, dimension, num_objecitves); 
        case 'uf1'
            mop = uf1(mop, dimension);
        case 'uf2'
            mop = uf2(mop, dimension);
        case 'uf3'
            mop = uf3(mop, dimension);
        case 'uf4'
            mop = uf4(mop, dimension);
        case 'uf5'
            mop = uf5(mop, dimension);
        case 'uf6'
            mop = uf6(mop, dimension);
        case 'uf7'
            mop = uf7(mop, dimension);
        case 'uf8'
            mop = uf8(mop, dimension);
        case 'uf9'
            mop = uf9(mop, dimension);
        case 'uf10'
            mop = uf10(mop, dimension);
        case 'mop1'
            mop = mop1(mop, dimension);
        case 'nas'
            mop = nas(mop, type);
        otherwise
            error('Undefined test problem name');
    end
end

%NAS function
function p = nas(p, type)
    load('nas\NATS_Bench1.mat','cifar10_data', 'cifar100_data', 'imagenet16_120_data', 'x_data');
    p.name = 'NAS';
    p.type = type;
    p.od = 2;
    p.pd = 6;
    p.domain = [ones(1, 6); 5 * ones(1, 6)];
    p.func = @evaluate;
    p.x = x_data;
    p.cifar10_data = cifar10_data;
    p.cifar100_data = cifar100_data;
    p.imagenet16_120 = imagenet16_120_data;
    function y = evaluate(x)
        x = round(x);
        [row_index, ~] = find(ismember(p.x, x, 'rows'), 1);
        switch p.type
            case 1
                y = p.cifar10_data(row_index, :);
            case 2
                y = p.cifar100_data(row_index, :);
            case 3
                y = p.imagenet16_120_data(row_index, :);
            otherwise
                error('Undefined function');
        end
    end
end

% KNO1 function generator
function p = kno1(p)
    p.name   = 'KNO1';
    p.od     = 2;
    p.pd     = 2;
    p.domain = [0, 3; 0, 3];
    p.func   = @evaluate;

    % KNO1 evaluation function
    function y = evaluate(x)
        y    = zeros(2, 1);
        c    = x(1) + x(2);
        f    = 9 - (3 * sin(2.5 * c^0.5) + 3 * sin(4 * c) + 5 * sin(2 * c + 2));
        g    = (pi / 2.0) * (x(1) - x(2) + 3.0) / 6.0;
        y(1) = 20 - (f * cos(g));
        y(2) = 20 - (f * sin(g));
    end
end

% ZDT1 function generator
function p = zdt1(p, dim)
    p.name   = 'ZDT1';
    p.pd     = dim;
    p.od     = 2;
    p.domain = [zeros(1, dim); ones(1, dim)];
    p.func   = @evaluate;
    p.pf     = @getPF;

    % ZDT1 evaluation function
    function y = evaluate(x)
        y1      = x(:, 1);
        su      = sum(x(:,2:end), 2);
        g       = 1 + 9 * su / (dim - 1);
        h       = 1 - sqrt(y1 ./ g);
        y2      = g .* h;
        y       = [y1, y2];
    end

    % PF
    function f = getPF
        f(:,1) = (0:1/9999:1)';
        f(:,2) = 1 - f(:,1).^0.5;
    end
end

% ZDT2 function generator
function p = zdt2(p, dim)
   p.name   = 'ZDT2';
    p.pd     = dim;
    p.od     = 2;
    p.domain = [zeros(1, dim); ones(1, dim)];
    p.func   = @evaluate;
    p.pf     = @getPF;

    % ZDT2 evaluation function
    function y = evaluate(x)
        y       = zeros(size(x, 1), p.od);
        y(:, 1) = x(:, 1);
        su      = sum(x, 2) - x(:, 1);
        g       = 1 + 9 * su / (dim - 1);
        y(:, 2) = g .* (1 - (y(:, 1) ./ g) .^ 2);
    end 

    % PF
    function f = getPF
        f(:,1) = (0:1/9999:1)';
        f(:,2) = 1 - f(:,1).^2;
    end
end

% ZDT3 function generator
function p = zdt3(p, dim)
    p.name   = 'ZDT3';
    p.pd     = dim;
    p.od     = 2;
    p.domain = [zeros(1, dim); ones(1, dim)];
    p.func   = @evaluate;
    p.pf     = @getPF;

    % ZDT3 evaluation function
    function y = evaluate(x)
        y       = zeros(size(x, 1), p.od);
        y(:, 1) = x(:, 1);
        su      = sum(x, 2) - x(:, 1);
        g       = 1 + 9 * su / (dim - 1);
        y(:, 2) = g .* (1 - sqrt(y(:, 1) ./ g) - (y(:, 1) ./ g) .* sin(10 * pi * y(:, 1)));
    end 

    % PF
    function f = getPF
        f(:,1) = (0:1/9999:1)';
        f(:,2) = 1 - f(:,1).^0.5 - f(:,1).*sin(10*pi*f(:,1));
        f      = f(NDSort(f,1)==1,:);
    end
end

% ZDT4 function generator
function p = zdt4(p, dim)
    p.name   = 'ZDT4';
    p.pd     = dim;
    p.od     = 2;
    p.domain = [0, -5 * ones(1, dim - 1); 1, 5 * ones(1, dim - 1)];
    p.func   = @evaluate;
    p.pf     = @getPF;

    % ZDT4 evaluation function
    function y = evaluate(x)
        y       = zeros(size(x, 1), p.od);
        y(:, 1) = x(:, 1);
        x_temp  = x(:, 2 : end);
        su      = x_temp .^ 2 - 10 * cos(4 * pi .* x_temp);
        g       = 1 + 10 * (dim - 1) + sum(su, 2);
        y(:, 2) = g .* (1 - sqrt(y(:, 1) ./ g));
    end

    % PF
    function f = getPF
        f(:,1) = (0:1/9999:1)';
        f(:,2) = 1 - f(:,1).^0.5;
    end
end

% ZDT6 function generator
function p = zdt6(p, dim)
    p.name   = 'ZDT6';
    p.pd     = dim;
    p.od     = 2;
    p.domain = [zeros(1, dim); ones(1, dim)];
    p.func   = @evaluate;
    p.pf     = @getPF;
    
    % ZDT6 evaluation function
    function y = evaluate(x)
        y       = zeros(size(x, 1), p.od);
        y(:, 1) = 1 - exp(-4 * x(:, 1)) .* (sin(6 * pi * x(:, 1)) .^ 6);
        su      = sum(x, 2) - x(:, 1);
        g       = 1 + 9 * (su / (dim - 1)) .^ 0.25;
        y(:, 2) = g .* (1 - (y(:, 1) ./ g) .^ 2);
    end

    % PF
    function f = getPF
        minf1  = 0.280775;
        f(:,1) = (minf1:(1-minf1)/9999:1)';
        f(:,2) = 1 - f(:,1).^2;
    end
end

% DTLZ suite assisted functions
function gx1 = gx1(P, nobj)
    k = size(P,2) - nobj + 1;
    xi = P(:,end-k+1:end);
    temp = (xi-0.5).^2 - cos(2*pi*(xi-0.5));
    temp = sum(temp, 2);
    g = 100 * (k+temp);
    gx1 = g + 1;
%     [popsize, nchrom] = size(P);
%     
%     g = zeros(popsize , 1);
%     for i = nobj : nchrom
%         temp = P(:, i) - 0.5;
%         g    = g + temp.*temp - cos(20 * pi .* temp);
%     end
%     g = 100 .* (nchrom - nobj + 1 + g);
%     gx1 = g + 1;
end

function gx2 = gx2(P, nobj)
    [popsize, nchrom] = size(P);
    
    g = zeros(popsize , 1);
    for i = nobj : nchrom
        temp = P(:, i) - 0.5;
        g    = g + temp .* temp;
    end
    gx2 = g + 1;
end

function gx6 = gx6(P, nobj)
    [popsize, nchrom] = size(P);

    g = zeros(popsize, 1);
    for i = nobj : nchrom
        temp = P(:, i);
        g    = g + temp .^ 0.1;
    end
    gx6 = g + 1;
end

function gx7 = gx7(P , nobj)
    [popsize, nchrom] = size(P);
    
    g = zeros(popsize, 1);
    for i = nobj : nchrom
        temp = P(:, i);
        g    = g + (9 ./ (nchrom - nobj + 1)) .* temp;
    end
    gx7 = g + 1;
end

function W = ReplicatePoint(SampleNum,M)
    if M > 1
        SampleNum = (ceil(SampleNum^(1/M)))^M;
        Gap       = 0:1/(SampleNum^(1/M)-1):1;
        eval(sprintf('[%s]=ndgrid(Gap);',sprintf('c%d,',1:M)))
        eval(sprintf('W=[%s];',sprintf('c%d(:),',1:M)))
    else
        W = (0:1/(SampleNum-1):1)';
    end
end

% DTLZ1 function generator
function p = dtlz1(p, dim, nobj)
    p.name   = 'DTLZ1';
    p.pd     = dim;
    p.od     = nobj;
    p.domain = [zeros(1, dim); ones(1, dim)];
    p.func   = @evaluate;
    p.pf     = @getPF;
 
    % DTLZ1 evaluation function
    function y = evaluate(x)
%         y = ones(size(x, 1), p.od);
%         
%         t   = gx1(x, p.od);
%         temp = ones(size(x, 1), p.od-1);
%         for i = 1:p.od-1
%             temp(:,i) = 1 - x(:,p.od-i);
%             for j = 1:p.od-i
%                 y(:,i) = x(:,j) .* y(:,i);
%             end
%         end
%         y(:,2:end) = y(:,2:end) .* temp;
%         y = y * 1/2 .* t;
          g = 100*(p.pd-p.od+1+sum((x(:,p.od:end)-0.5).^2-cos(2.*pi.*(x(:,p.od:end)-0.5)),2));
          y = 0.5*repmat(1+g,1,p.od).*fliplr(cumprod([ones(size(x,1),1),x(:,1:p.od-1)],2)).*[ones(size(x,1),1),1-x(:,p.od-1:-1:1)];
    end

    % PF
    function f = getPF
        f = UniformPoint(10000,p.od)/2;
    end
end

% DTLZ2 function generator
function p = dtlz2(p, dim, nobj)
    p.name   = 'DTLZ2';
    p.pd     = dim;
    p.od     = nobj;
    p.domain = [zeros(1, dim); ones(1, dim)];
    p.func   = @evaluate;
    p.pf     = @getPF;

    % DTLZ2 evaluation function
    function y = evaluate(x)
        y = zeros(size(x, 1), p.od);

        t   = gx2(x , p.od);
        sum = t;
        for i = 1 : p.od - 1
            sum = sum .* cos(x(: , i) .* pi ./ 2.0);
        end
        y(: , 1) = sum;

        for i = 2 : p.od
            sum = t;
            for j = 1 : p.od - i
                sum = sum .* cos(x(: , j) .* pi ./ 2.0);
            end
            sum = sum .* sin(x(: , p.od - i + 1) .* pi ./ 2.0);
            y(: , i) = sum;
        end
    end

    % PF
    function f = getPF
        f = UniformPoint(10000,p.od)/2;
        f = f./repmat(sqrt(sum(f.^2,2)),1,p.od);
    end
end

% DTLZ3 function generator
function p = dtlz3(p, dim, nobj)
    p.name   = 'DTLZ3';
    p.pd     = dim;
    p.od     = nobj;
    p.domain = [zeros(1, dim); ones(1, dim)];
    p.func   = @evaluate;
    p.pf     = @getPF;

    % DTLZ3 evaluation function
    function y = evaluate(x)
%         y = zeros(size(x, 1), p.od);
% 
%         t = gx1(x, nobj);
%         sum = t ;
%         for i = 1 : p.od - 1
%             sum = sum .* cos(x(: , i) .* pi ./ 2.0);
%         end
%         y(: , 1) = sum ;
% 
%         for i = 2 : p.od
%             sum = t ;
%             for j = 1 : p.od - i
%                 sum = sum .* cos(x(: , j) .* pi ./ 2.0);
%             end
%             sum = sum .* sin(x(: , p.od - i + 1) .* pi ./ 2.0) ;
%             y(: , i) = sum;
        g      = (100)*(p.pd-p.od+1+sum((x(:,p.od:end)-0.5).^2-cos(2.*pi.*(x(:,p.od:end)-0.5)),2));
       y      = repmat(1+g,1,p.od).*fliplr(cumprod([ones(size(x,1),1),cos(x(:,1:p.od-1)*pi/2)],2)).*[ones(size(x,1),1),sin(x(:,p.od-1:-1:1)*pi/2)];
    end

    % PF
    function f = getPF
        f = UniformPoint(10000,p.od)/2;
        f = f./repmat(sqrt(sum(f.^2,2)),1,p.od);
    end
end

% DTLZ4 function generator
function p = dtlz4(p, dim, nobj)
    p.name   = 'DTLZ4';
    p.pd     = dim;
    p.od     = nobj;
    p.domain = [zeros(1, dim); ones(1, dim)];
    p.func   = @evaluate;
    p.pf     = @getPF;

    % DTLZ4 evaluation function
    function y = evaluate(x)
        y = zeros(size(x, 1), p.od);

        t = gx2(x, nobj);
        sum = t ;
        a   = 100;
        x(:, [1, p.od - 1]) = x(:, [1, p.od - 1]) .^ a;
        for i = 1 : p.od - 1
            sum = sum .* cos(x(: , i) .* pi ./ 2.0);
        end
        y(: , 1) = sum ;

        for i = 2 : p.od
            sum = t ;
            for j = 1 : p.od - i
                sum = sum .* cos(x(: , j) .* pi ./ 2.0);
            end
            sum = sum .* sin(x(:, p.od - i + 1) .* pi ./ 2.0) ;
            y(: , i) = sum;
        end
    end

    % PF
    function f = getPF
        f = UniformPoint(10000,p.od)/2;
        f = f./repmat(sqrt(sum(f.^2,2)),1,p.od);
    end
end

% DTLZ5 function generator (low dimension)
function p = dtlz5(p, dim, nobj)
    p.name   = 'DTLZ5';
    p.pd     = dim;
    p.od     = nobj;
    p.domain = [zeros(1, dim); ones(1, dim)];
    p.func   = @evaluate;
    p.pf     = @getPF;

    % DTLZ5 evaluation function
    function y = evaluate(x)
        y = zeros(size(x, 1), p.od);

        t = gx2(x, p.od) - 1;
        for i = 2 : p.od - 1
            x(:, i) = pi ./ (4 .* (1 + t)) .* (1 + 2 .* t .* x(:, i));
        end

        sum = t + 1;
        for i = 2 : p.od - 1
            sum = sum .* cos(x(: , i));
        end
        sum = sum .* cos(x(:, 1) .* pi ./ 2.0);
        y(:, 1) = sum;

        for i = 2 : p.od
            sum = t + 1;
            for j = 2 : p.od - i
                sum = sum .* cos(x(: , j));
            end
            if i == p.od
                sum = sum .* sin(x(:, 1) .* pi ./ 2.0);
            else
                sum = sum .* sin(x(:, p.od - i + 1));
                sum = sum .* cos(x(:, 1) .* pi ./ 2.0);
            end
            y(: , i) = sum;
        end
    end

    % PF
    function f = getPF
        f = [0:1/9999:1;1:-1/9999:0]';
        f = f./repmat(sqrt(sum(f.^2,2)),1,size(f,2));
        f = [f(:,ones(1,p.od-2)),f];
        f = f./sqrt(2).^repmat([p.od-2,p.od-2:-1:0],size(f,1),1);
    end
end

% DLTZ5i function generator (high dimension)
function p = dtlz5i(p, dim, nobj)
    p.name   = 'DTLZ5i';
    p.pd     = dim;
    p.od     = nobj;
    p.domain = [zeros(1, dim); ones(1, dim)];
    p.func   = @evaluate;

    % DTLZ5i evaluation function
    function y = evaluate(x)
        y = zeros(size(x, 1), p.od);

        [t, x] = gx5i(x, p.od);
        sum = t;

        for i = 1 : p.od - 1
            sum = sum .* cos(x(:, i));
        end
        y(:, 1) = sum;

        for i = 2 : p.od
            sum = t;
            for j = 1 : p.od - i
                sum = sum .* cos(x(:, j));
            end
            sum = sum .* sin(x(:, p.od - i + 1));
            y(:, i) = sum;
        end
    end
end

% DTLZ6 function generator
function p = dtlz6(p, dim, nobj)
    p.name   = 'DTLZ6';
    p.pd     = dim;
    p.od     = nobj;
    p.domain = [zeros(1, dim); ones(1, dim)];
    p.func   = @evaluate;
    p.pf     = @getPF;

    % DTLZ6 evaluation function
    function y = evaluate(x)
        y = zeros(size(x, 1), p.od);

        t = gx6(x, p.od) - 1;
        for i = 2 : p.od - 1
            x(:, i) = pi ./ (4 .* (1 + t)) .* (1 + 2 .* t .* x(:, i));
        end

        sum = t + 1;
        for i = 2 : p.od - 1
            sum = sum .* cos(x(: , i));
        end
        sum = sum .* cos(x(:, 1) .* pi ./ 2.0);
        y(:, 1) = sum;

        for i = 2 : p.od
            sum = t + 1;
            for j = 2 : p.od - i
                sum = sum .* cos(x(: , j));
            end
            if i == p.od
                sum = sum .* sin(x(:, 1) .* pi ./ 2.0);
            else
                sum = sum .* sin(x(:, p.od - i + 1));
                sum = sum .* cos(x(:, 1) .* pi ./ 2.0);
            end
            y(: , i) = sum;
        end
    end

    % PF
    function f = getPF
        f = [0:1/9999:1;1:-1/9999:0]';
        f = f./repmat(sqrt(sum(f.^2,2)),1,size(f,2));
        f = [f(:,ones(1,p.od-2)),f];
        f = f./sqrt(2).^repmat([p.od-2,p.od-2:-1:0],size(f,1),1);
    end
end

% DTLZ7 function generator
function p = dtlz7(p, dim, nobj)
    p.name   = 'DTLZ7';
    p.pd     = dim;
    p.od     = nobj;
    p.domain = [zeros(1, dim); ones(1, dim)];
    p.func   = @evaluate;
    p.pf     = @getPF;

    % DTLZ7 evaluation function
    function y = evaluate(x)
        [popsize, ~] = size(x);
        y = zeros(popsize, p.od);

        t = gx7(x, p.od);

        for i = 1 : p.od - 1
            y(:, i) = x(:, i);
        end

        sum = t;
        temp = zeros(popsize, 1);

        for i = 1 : p.od - 1
            temp = temp + (y(:, i) ./ (1 + sum)) .* (1 + sin(3 .* pi .* y(:, i)));
        end
        temp = p.od - temp;
        y(:, p.od) = (sum + 1) .* temp;
    end

    % PF
    function f = getPF
        interval     = [0,0.251412,0.631627,0.859401];
        median       = (interval(2)-interval(1))/(interval(4)-interval(3)+interval(2)-interval(1));
        X            = ReplicatePoint(10000,p.od-1);
        X(X<=median) = X(X<=median)*(interval(2)-interval(1))/median+interval(1);
        X(X>median)  = (X(X>median)-median)*(interval(4)-interval(3))/(1-median)+interval(3);
        f            = [X,2*(p.od-sum(X/2.*(1+sin(3*pi.*X)),2))];
    end
end

% UF1 function generator
function p = uf1(p, dim)
    p.name   = 'UF1';
    p.pd     = dim;
    p.od     = 2;
    p.domain = [0, -1 * ones(1, dim - 1); ones(1, dim)];
    p.func   = @evaluate;
    
    % UF1 evaluation function
    function y = evaluate(x)
        x               = x';
        [dim, num]      = size(x);
        tmp             = zeros(dim,num);
        tmp(2 : dim, :) = (x(2 : dim, :) - sin(6.0 * pi * repmat(x(1, :), [dim - 1, 1]) + pi / dim * repmat((2 : dim)', [1, num]))) .^ 2;
        tmp1            = sum(tmp(3 : 2 : dim, :));  % odd index
        tmp2            = sum(tmp(2 : 2 : dim, :));  % even index
        y(1, :)         = x(1, :) + 2.0 * tmp1 / size(3 : 2 : dim, 2);
        y(2, :)         = 1.0 - sqrt(x(1, :)) + 2.0 * tmp2 / size(2 : 2 : dim, 2);
        y               = y';
        clear tmp;
    end
end

% UF2 function generator
function p = uf2(p, dim)
    p.name   = 'UF2';
    p.pd     = dim;
    p.od     = 2;
    p.domain = [0, -1 * ones(1, dim - 1); ones(1, dim)];
    p.func   = @evaluate;
    
    % UF2 evaluation function
    function y = evaluate(x)
        x               = x';
        [dim, num]      = size(x);
        X1              = repmat(x(1, :), [dim - 1, 1]);
        A               = 6 * pi * X1 + pi / dim * repmat((2 : dim)', [1, num]);
        tmp             = zeros(dim, num);
        tmp(2 : dim, :) = (x(2 : dim, :) - 0.3 * X1 .* (X1 .* cos(4.0 * A) + 2.0) .* cos(A)) .^ 2;
        tmp1            = sum(tmp(3 : 2 : dim, :));  % odd index
        tmp(2 : dim, :) = (x(2 : dim, :) - 0.3 * X1 .* (X1 .* cos(4.0 * A) + 2.0) .* sin(A)) .^ 2;
        tmp2            = sum(tmp(2 : 2 : dim, :));  % even index
        y(1, :)         = x(1, :) + 2.0 * tmp1 / size(3 : 2 : dim, 2);
        y(2, :)         = 1.0 - sqrt(x(1, :)) + 2.0 * tmp2 / size(2 : 2 : dim, 2);
        y               = y';
        clear X1 A tmp;
    end
end

% UF3 function generator
function p = uf3(p, dim)
    p.name   = 'UF3';
    p.pd     = dim;
    p.od     = 2;
    p.domain = [zeros(1, dim); ones(1, dim)];
    p.func   = @evaluate;
    
    % UF3 evaluation function
    function y = evaluate(x)
        x                = x';
        [dim, num]       = size(x);
        Y                = zeros(dim, num);
        Y(2 : dim, :)    = x(2 : dim, :) - repmat(x(1, :), [dim - 1, 1]) .^ (0.5 + 1.5 * (repmat((2 : dim)', [1, num]) - 2.0) / (dim - 2.0));
        tmp1             = zeros(dim, num);
        tmp1(2 : dim, :) = Y(2 : dim, :) .^ 2;
        tmp2             = zeros(dim, num);
        tmp2(2 : dim, :) = cos(20.0 * pi * Y(2 : dim, :) ./ sqrt(repmat((2 : dim)', [1, num])));
        tmp11            = 4.0 * sum(tmp1(3 : 2 : dim, :)) - 2.0 * prod(tmp2(3 : 2 : dim, :)) + 2.0;  % odd index
        tmp21            = 4.0 * sum(tmp1(2 : 2 : dim, :)) - 2.0 * prod(tmp2(2 : 2 : dim, :)) + 2.0;  % even index
        y(1, :)          = x(1, :)             + 2.0 * tmp11 / size(3 : 2 : dim, 2);
        y(2, :)          = 1.0 - sqrt(x(1, :)) + 2.0 * tmp21 / size(2 : 2 : dim, 2);
        y                = y';
        clear Y tmp1 tmp2;
    end
end

% UF4 function generator
function p = uf4(p, dim)
    p.name   = 'UF4';
    p.pd     = dim;
    p.od     = 2;
    p.domain = [0, -2 * ones(1, dim - 1); 1, 2 * ones(1, dim - 1)];
    p.func   = @evaluate;
    
    % UF4 evaluation function
    function y = evaluate(x)
        x              = x';
        [dim, num]     = size(x);
        Y              = zeros(dim, num);
        Y(2 : dim, :)  = x(2 : dim, :) - sin(6.0 * pi * repmat(x(1, :),[dim - 1, 1]) + pi / dim * repmat((2 : dim)', [1, num]));
        H              = zeros(dim, num);
        H(2 : dim, :)  = abs(Y(2 : dim, :)) ./ (1.0 + exp(2.0 * abs(Y(2 : dim, :))));
        tmp1           = sum(H(3 : 2 : dim, :));  % odd index
        tmp2           = sum(H(2 : 2 : dim, :));  % even index
        y(1, :)        = x(1, :) + 2.0 * tmp1 / size(3 : 2 : dim, 2);
        y(2, :)        = 1.0 - x(1, :) .^ 2 + 2.0 * tmp2 / size(2 : 2 : dim, 2);
        y              = y';
        clear Y H;
    end
end

% UF5 function generator
function p = uf5(p, dim)
    p.name   = 'UF5';
    p.pd     = dim;
    p.od     = 2;
    p.domain = [0, -1 * ones(1, dim - 1); ones(1, dim)];
    p.func   = @evaluate;
    
    % UF5 evaluation function
    function y = evaluate(x)
        x              = x';
        N              = 10.0;
        E              = 0.1;
        [dim, num]     = size(x);
        Y              = zeros(dim, num);
        Y(2 : dim, :)  = x(2 : dim, :) - sin(6.0 * pi * repmat(x(1, :), [dim - 1, 1]) + pi / dim * repmat((2 : dim)', [1, num]));
        H              = zeros(dim, num);
        H(2 : dim, :)  = 2.0 * Y(2 : dim, :) .^ 2 - cos(4.0 * pi * Y(2 : dim, :)) + 1.0;
        tmp1           = sum(H(3 : 2 : dim, :));  % odd index
        tmp2           = sum(H(2 : 2 : dim, :));  % even index
        tmp            = (0.5 / N + E) * abs(sin(2.0 * N * pi * x(1, :)));
        y(1,:)         = x(1, :) + tmp + 2.0 * tmp1 / size(3 : 2 : dim, 2);
        y(2,:)         = 1.0 - x(1, :)+ tmp + 2.0 * tmp2 / size(2 : 2 : dim, 2);
        y              = y';
        clear Y H;
    end
end

% UF6 function generator
function p = uf6(p, dim)
    p.name   = 'UF6';
    p.pd     = dim;
    p.od     = 2;
    p.domain = [0, -1 * ones(1, dim - 1); ones(1, dim)];
    p.func   = @evaluate;
    
    % UF6 evaluation function
    function y = evaluate(x)
        x                = x';
        N                = 2.0;
        E                = 0.1;
        [dim, num]       = size(x);
        Y                = zeros(dim, num);
        Y(2 : dim, :)    = x(2 : dim, :) - sin(6.0 * pi * repmat(x(1, :), [dim - 1, 1]) + pi / dim * repmat((2 : dim)', [1, num]));
        tmp1             = zeros(dim, num);
        tmp1(2 : dim, :) = Y(2 : dim, :) .^ 2;
        tmp2             = zeros(dim, num);
        tmp2(2 : dim, :) = cos(20.0 * pi * Y(2 : dim, :) ./ sqrt(repmat((2 : dim)', [1, num])));
        tmp11            = 4.0 * sum(tmp1(3 : 2 : dim, :)) - 2.0 * prod(tmp2(3 : 2 : dim, :)) + 2.0;  % odd index
        tmp21            = 4.0 * sum(tmp1(2 : 2 : dim, :)) - 2.0 * prod(tmp2(2 : 2 : dim, :)) + 2.0;  % even index
        tmp              = max(0, (1.0 / N + 2.0 * E)*sin(2.0 * N * pi * x(1, :)));
        y(1, :)          = x(1, :) + tmp + 2.0 * tmp11 / size(3 : 2 : dim, 2);
        y(2, :)          = 1.0 - x(1, :) + tmp + 2.0 * tmp21 / size(2 : 2 : dim, 2);
        y                = y';
        clear Y tmp1 tmp2;
    end
end

% UF7 function generator
function p = uf7(p, dim)
    p.name   = 'UF7';
    p.pd     = dim;
    p.od     = 2;
    p.domain = [0, -1 * ones(1, dim - 1); ones(1, dim)];
    p.func   = @evaluate;
    
    % UF7 evaluation function
    function y = evaluate(x)
        x             = x';
        [dim, num]    = size(x);
        Y             = zeros(dim, num);
        Y(2 : dim, :) = (x(2 : dim, :) - sin(6.0 * pi * repmat(x(1, :), [dim - 1, 1]) + pi / dim * repmat((2 : dim)', [1, num]))) .^ 2;
        tmp1          = sum(Y(3 : 2 : dim, :));  % odd index
        tmp2          = sum(Y(2 : 2 : dim, :));  % even index
        tmp           = (x(1, :)) .^ 0.2;
        y(1, :)       = tmp + 2.0 * tmp1 / size(3 : 2 : dim, 2);
        y(2, :)       = 1.0 - tmp + 2.0 * tmp2 / size(2 : 2 : dim, 2);
        y             = y';
        clear Y;
    end
end

% UF8 function generator
function p = uf8(p, dim)
    p.name   = 'UF8';
    p.pd     = dim;
    p.od     = 3;
    p.domain = [0, 0, -2 * ones(1, dim - 2); 1, 1, 2 *ones(1, dim - 2)];
    p.func   = @evaluate;
    
    % UF8 evaluation function
    function y = evaluate(x)
        x             = x';
        [dim, num]    = size(x);
        Y             = zeros(dim, num);
        Y(3 : dim, :) = (x(3 : dim, :) - 2.0 * repmat(x(2, :), [dim - 2, 1]) .* sin(2.0 * pi * repmat(x(1, :), [dim - 2, 1]) + pi / dim * repmat((3 : dim)', [1, num]))) .^ 2;
        tmp1          = sum(Y(4 : 3 : dim, :));  % j-1 = 3*k
        tmp2          = sum(Y(5 : 3 : dim, :));  % j-2 = 3*k
        tmp3          = sum(Y(3 : 3 : dim, :));  % j-0 = 3*k
        y(1, :)       = cos(0.5 * pi * x(1, :)) .* cos(0.5 * pi * x(2, :)) + 2.0 * tmp1 / size(4 : 3 : dim, 2);
        y(2, :)       = cos(0.5 * pi * x(1, :)) .* sin(0.5 * pi * x(2, :)) + 2.0 * tmp2 / size(5 : 3 : dim, 2);
        y(3, :)       = sin(0.5 * pi * x(1, :)) + 2.0 * tmp3 / size(3 : 3 : dim, 2);
        y             = y';
        clear Y;
    end
end

% UF9 function generator
function p = uf9(p, dim)
    p.name   = 'UF9';
    p.pd     = dim;
    p.od     = 3;
    p.domain = [0, 0, -2 * ones(1, dim - 2); 1, 1, 2 *ones(1, dim - 2)];
    p.func   = @evaluate;
    
    % UF9 evaluation function
    function y = evaluate(x)
        x             = x';
        E             = 0.1;
        [dim, num]    = size(x);
        Y             = zeros(dim, num);
        Y(3 : dim, :) = (x(3 : dim, :) - 2.0 * repmat(x(2, :), [dim - 2, 1]) .* sin(2.0 * pi * repmat(x(1, :), [dim - 2, 1]) + pi / dim * repmat((3 : dim)', [1, num]))) .^ 2;
        tmp1          = sum(Y(4 : 3 : dim, :));  % j-1 = 3*k
        tmp2          = sum(Y(5 : 3 : dim, :));  % j-2 = 3*k
        tmp3          = sum(Y(3 : 3 : dim, :));  % j-0 = 3*k
        tmp           = max(0,(1.0+E)*(1-4.0*(2.0*x(1,:)-1).^2));
        y(1, :)       = 0.5 * (tmp + 2 * x(1, :)) .* x(2, :) + 2.0 * tmp1 / size(4 : 3 : dim, 2);
        y(2, :)       = 0.5 * (tmp - 2 * x(1, :) + 2.0) .* x(2, :) + 2.0*tmp2/size(5 : 3 : dim, 2);
        y(3, :)       = 1 - x(2, :) + 2.0 * tmp3 / size(3 : 3 : dim, 2);
        y             = y';
        clear Y;
    end
end

% UF10 function generator
function p = uf10(p, dim)
    p.name   = 'UF10';
    p.pd     = dim;
    p.od     = 3;
    p.domain = [0, 0, -2 * ones(1, dim - 2); 1, 1, 2 *ones(1, dim - 2)];
    p.func   = @evaluate;
    
    % UF9 evaluation function
    function y = evaluate(x)
        x             = x';
        [dim, num]    = size(x);
        Y             = zeros(dim, num);
        Y(3 : dim, :) = x(3 : dim, :) - 2.0 * repmat(x(2, :), [dim - 2, 1]) .* sin(2.0 * pi * repmat(x(1, :), [dim - 2, 1]) + pi / dim * repmat((3 : dim)', [1, num]));
        H             = zeros(dim, num);
        H(3 : dim, :) = 4.0 * Y(3 : dim, :) .^ 2 - cos(8.0 * pi * Y(3 : dim, :)) + 1.0;
        tmp1          = sum(H(4 : 3 : dim, :));  % j-1 = 3*k
        tmp2          = sum(H(5 : 3 : dim, :));  % j-2 = 3*k
        tmp3          = sum(H(3 : 3 : dim, :));  % j-0 = 3*k
        y(1, :)       = cos(0.5 * pi * x(1, :)) .* cos(0.5 * pi * x(2, :)) + 2.0 * tmp1 / size(4 : 3 : dim, 2);
        y(2, :)       = cos(0.5 * pi * x(1, :)) .* sin(0.5 * pi * x(2, :)) + 2.0 * tmp2 / size(5 : 3 : dim, 2);
        y(3, :)       = sin(0.5 * pi * x(1, :)) + 2.0 * tmp3 / size(3 : 3 : dim, 2);
        y             = y';
        clear Y H;
    end
end

function t = ti(x)
    [~, nchrom] = size(x);
    t = x - repmat(sin(0.5 * pi * x(:, 1)) , 1 , nchrom);
end

function g = gx(x)
    [popsize, nchrom] = size(x);
    t                 = ti(x);

    sum = zeros(popsize, 1);
    for i = 2 : nchrom
        sum = sum + ((0 - 0.9) * t(:, i) .* t(:, i) + abs(t(:, i)) .^ 0.6);
    end

    g = 2 * sin(pi * x(:, 1)) .* sum;
end

% MOP1 function generator
function p = mop1(p, dim)
    p.name   = 'MOP1';
    p.pd     = dim;
    p.od     = 2;
    p.domain = [zeros(1, dim); ones(1, dim)];
    p.func   = @evaluate;
    
    % MOP1 evaluation function
    function y = evaluate(x)
        g = gx(x);

        y(:, 1) = (1 + g) .* x(:, 1);
        y(:, 2) = (1 + g) .* (1 - sqrt(x(:, 1)));
    end  
end