function CubeData=generate3ComponentData(imA,imB,imC,tau1,tau2,tau3,fileName)
% create ome tiff decay data for testing with decay 

if(nargin==6)
    saveFile=0;
elseif(nargin==7)
    saveFile=1;

end

if(~((size(imA)==size(imB))==(size(imA)==size(imC))))
    fprintf('images should be equal in dimension.. exiting')
    return;
end

[rowSize, colSizes]=size(imA);
decayBins=300;


CubeData=double(zeros(rowSize,colSizes,decayBins));% (x,y,t)


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


% real noise



NoiseModel=normrnd(1600,10,size(CubeData)); % normal noise

CubeData=CubeData+NoiseModel;
%


%         
        
if(saveFile==1)

    delete(fileName);
    bfsave(CubeData, fileName,'dimensionOrder','XYTZC')

end

