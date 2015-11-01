function [ labels ] = supportAcc( rY, tY )

labels = (rY(:) ~= 0) == (tY(:) ~= 0);
labels = sum(labels);
labels = labels/numel(rY);

end

