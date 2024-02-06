function pop = assignV(pos, obj, pop)

    for i = 1 : size(pos, 1)
        pop(i).Position = pos(i,:);
        pop(i).Cost = obj(i,:);
    end

end
