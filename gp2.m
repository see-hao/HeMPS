tic
x = lhsdesign(100, 1);
y = x.^2;
model = fitrgp(x, y);
toc;
time = toc;