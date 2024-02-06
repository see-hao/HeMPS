clc
clear

n = 200;
d = 8;
m = 3;
p = 'dtlz1';
problem = testmop(p, d, m);

x = lhsdesign(n, problem.pd);
ys = problem.func(x);
weight = rand(1, problem.od);
weight = weight / sum(weight);

m = size(ys, 2);
    for i = 1 : m
        tmp(:, i) = ys(:,i) * weight(i);
    end
y = max(tmp, [], 2);

model = fitrgp(x, y);
save('e1_model/DTLZ1_3_8_1', 'model');