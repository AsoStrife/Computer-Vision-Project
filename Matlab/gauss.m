function G=gauss(x,sigma)

%G = exp(-x.^2/(2*sigma^2))/(sqrt(2*pi)*sigma);
G = exp(-x.^2/(2*sigma^2));
G=G/sum(sum(G));

end
