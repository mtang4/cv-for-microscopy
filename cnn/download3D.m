trainSize = 1000;  % change training size as necessary

imAt=double(imread('singleMito.tif'));
imBt=double(imread('singleActin.tif'));
imCt=double(imread('singleNuc.tif'));
imA=imresize(imAt,[256 256]);
imB=imresize(imBt,[256 256]);
imC=imresize(imCt,[256 256]);

d=(size(imA)==size(imB))==(size(imA)==size(imC));

params = zeros(6,trainSize);
angles = [0,90,180,270];
for i=1:trainSize
    tau1 = 5*rand(1);
    tau2 = 5*rand(1);
    tau3 = 5*rand(1);
    
    randIndA = randperm(length(angles),1);
    randIndB = randperm(length(angles),1);
    randIndC = randperm(length(angles),1);
    thetaA = angles(randIndA);
    thetaB = angles(randIndB);
    thetaC = angles(randIndC);

    params(:,i)=[imA(1); imB(1); imC(1); tau1; tau2; tau3];
    data=single(generate3ComponentData(imrotate(imA,thetaA),imrotate(imB,thetaB),imrotate(imC,thetaC),tau1,tau2,tau3));
   
    file = sprintf('image%d.mat', i);
    save(file, 'data')
end
save('train_labels.mat', 'params'). % save parameters to create labels for training set

% function definition
function CubeData=generate3ComponentData(imA,imB,imC,tau1,tau2,tau3)

if(~((size(imA)==size(imB))==(size(imA)==size(imC))))
    fprintf('images should be equal in dimension.. exiting')
    return;
end

[rowSize, colSizes]=size(imA);
decayBins=300;

CubeData=double(zeros(rowSize,colSizes,decayBins));
timePoints=0.1:0.1:30;

 for i=1:rowSize
    for j=1:colSizes
        data=double(imA(i,j))*(exp(-(normrnd(tau1,tau1*0.04))*timePoints))+...
            double(imB(i,j))*(exp(-(normrnd(tau2,tau2*0.04))*timePoints))+1*rand(1,decayBins)+...
            double(imC(i,j))*(exp(-(normrnd(tau3,tau3*0.04))*timePoints));
       
        if(~isnan(sum(data)))
            CubeData(i,j,:)=data;
        end
    end
 end

% NoiseModel=normrnd(1600,10,size(CubeData)); % normal noise
% CubeData=CubeData+NoiseModel;

end
