function imgResult=gaussianfilter(img,sigma,shift)

if max(size(sigma))==1, sigma=[sigma sigma]; end
if nargin<3 || isempty(shift), shift = 0; end
if max(size(shift)) == 1, shift=[shift shift]; end

%x = [floor(-3.0*sigma+0.5):floor(3.0*sigma+0.5)];
Gx = gauss([ceil(-3.0*sigma(1)-0.5-shift(1)):floor(3.0*sigma(1)+0.5-shift(1))],sigma(1));
Gy = gauss([ceil(-3.0*sigma(2)-0.5-shift(2)):floor(3.0*sigma(2)+0.5-shift(2))],sigma(2));
imgTmp    = conv2(img,Gx,'same');
imgResult = conv2(imgTmp,Gy','same');

end
