function [obj] = getObjAPGnc(loss, s, lambda, theta, regType)

obj = 0.5*sum(loss.^2);
obj = obj + funRegC(s, length(s), lambda, theta, regType);

end