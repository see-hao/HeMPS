function smodels = select_model(o, src_models, num)
    %num代表选几个模型
    %生成随机100个点
    x = create_x(100, size(o.tar_model.X, 2));
    kmodel = kpca_train(x, o.option);
    %得到目标模型产生的y
    y = o.tar_model.predict(x);
    sz = size(src_models, 2);
    for i = 1 : sz
        ys = src_models{1, i}.predict(kmodel.mappedX);
        error = sum((y - ys).^2);
        diff(1, i) = error;
    end
    [diff, index] = sort(diff);
    for i = 1 : num
        smodels{1, i} = src_models{1, index(i)};
    end
end

function x = create_x(n, od)
    x = zeros(n, od);
    for i = 1:n
       x(i,:) = randi([1, 5], 1, od);
    end
end