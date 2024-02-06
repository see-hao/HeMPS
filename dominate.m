function a = dominate(p, q)
    objs = size(p.Cost, 2);
    a = 0;
    for i = 1 : objs
        if p.Cost(1, i) < q.Cost(1, i)
            a = a + 1;
        else a = a - 1;
        end
    end
    if a == objs
        a = 1;
    else a = 0;
    end
end