function [ delta ] = getMaxOverk( obj, q )

if(isempty(obj))
    delta = inf;
else
    if(length(obj) - q <= 1)
        delta = max(obj);
    else
        delta = max(obj(length(obj) - q: length(obj)));
    end
end

end

