function [dataOut,errSurface,errSurfaceMin,errSurfaceMax] = shadingCorrection(dataIn,numScales)
%function [dataOut,errSurface,errSurfaceMin,errSurfaceMax] = shadingCorrection(dataIn)
%
%-------- this function corrects the shading from images -----------------------------
%-------------------------------------------------------------------------------------
%------  Author :   Constantino Carlos Reyes-Aldasoro                       ----------
%------             Research Fellow  Sheffield University                   ----------
%------             http://carlos-reyes.staff.shef.ac.uk                    ----------
%------  5 February 2009                                   ---------------------------
%-------------------------------------------------------------------------------------
% A function to correct the shading (inhomogeneous background intensities) of images.
% The function estimates the background from an envelope of the data and then substracts
% it from the data to return an image without shading. The only assumption is that the
% objects are either darker or brigther than the background (but not both)

% input data:       dataIn          : an image with cells vessels or any other kind of objects
% output data:      dataOut         : image with a uniform background
%                   errSurface      : the shading surface
%                   errSurfaceMin   : the lower envelope
%                   errSurfaceMax   : the higher envelope

% Earlier version of the function were colourBackgroundequalise*.m

%% Parse input data
%------ no input data is received, error -------------------------
if nargin <1;     help shadingCorrection; dataOut=[]; return;  end;


%-------- regular size check and determination of first zoom region
[rows,cols,levs]=size(dataIn);
if ~isa(dataIn,'double'); 
    dataIn=double(dataIn); 
    reconvertTo255 =1;
else
    reconvertTo255 =0;
end

%% Define parameters
if ~exist('numScales','var')
    numScales                           = 55;       % maximum distance of analysis
end
stopCriterion                       = 0.05;     % stop criterion difference between steps
% 
% % initialise parameters
% y_scaleMax(rows,cols,numScales)     = 0;
% y_scaleMin(rows,cols,numScales)     = 0;
% dataOut(rows,cols,levs)             = 0;        %#ok<NASGU>
% sizeFR_S                            = 3;        % size of a Gaussian filter to remove noise
% sizeFC_S                            = sizeFR_S;
% filtG_small                         = gaussF(sizeFR_S,sizeFC_S,1); %define the filter


if levs>1
    % initialise parameters
    %y_scaleMax(rows,cols,numScales)     = 0;
    %y_scaleMin(rows,cols,numScales)     = 0;
    dataOut(rows,cols,levs)             = 0;        %#ok<NASGU>
    %sizeFR_S                            = 3;        % size of a Gaussian filter to remove noise
    %sizeFC_S                            = sizeFR_S;
    %filtG_small                         = gaussF(sizeFR_S,sizeFC_S,1); %define the filter

    %in case the data is 3D (a colour image or a volume) subsample to reduce complexity and call same
    %function for each individual channel (level)
    subSampl=1;
    if nargout==4
        for currLev=1:levs
            %[dataOut_S(:,:,currLev),errSurface_S(:,:,currLev)]=shadingCorrection(reduceu(dataIn(:,:,currLev),subSampl/2)); %#ok<AGROW>
            [dataOut_S(:,:,currLev),errSurface_S(:,:,currLev),errSurfaceMin_S(:,:,currLev),errSurfaceMax_S(:,:,currLev)]=shadingCorrection(dataIn(1:subSampl:end,1:subSampl:end,currLev)); %#ok<AGROW>
            %errSurface(:,:,currLev) = expandu(errSurface_S(:,:,currLev),subSampl/2);         %#ok<AGROW>
            %errSurfaceMax(:,:,currLev) = expandu(errSurfaceMax_S(:,:,currLev),subSampl/2);         %#ok<AGROW>
            %errSurfaceMin(:,:,currLev) = expandu(errSurfaceMin_S(:,:,currLev),subSampl/2);         %#ok<AGROW>
        end
        dataOut=dataOut_S;
        errSurface=errSurface_S;
        errSurfaceMax=errSurfaceMax_S;
        errSurfaceMin=errSurfaceMin_S;
        %dataOut                     = dataIn - errSurface(1:rows,1:cols,:);
        errSurfaceMax               = errSurfaceMax(1:rows,1:cols,:);
        errSurfaceMin               = errSurfaceMin(1:rows,1:cols,:);
        dataOut(dataOut>255)        = 255;
        dataOut(dataOut<0)          = 0;
        dataOut                     = (dataOut-min(dataOut(:)));
        dataOut                     = 255*(dataOut/max(dataOut(:)));

    else
        for currLev=1:levs
            %[dataOut_S(:,:,currLev),errSurface_S(:,:,currLev)]=shadingCorrection(reduceu(dataIn(:,:,currLev),subSampl/2)); %#ok<AGROW>
            [dataOut_S(:,:,currLev),errSurface_S(:,:,currLev),errSurfaceMin_S(:,:,currLev),errSurfaceMax_S(:,:,currLev)]=shadingCorrection(dataIn(1:subSampl:end,1:subSampl:end,currLev)); %#ok<AGROW>
            %errSurface(:,:,currLev) = expandu(errSurface_S(:,:,currLev),subSampl/2);         %#ok<AGROW>
        end
        %dataOut = dataIn - errSurface(1:rows,1:cols,:);
        dataOut=dataOut_S;
        errSurface=errSurface_S;
        dataOut(dataOut>255)=255;
        dataOut(dataOut<0)=0;
        %dataOut                     = (dataOut-min(dataOut(:)));
        %dataOut                     = 255*(dataOut/max(dataOut(:)));
    end
else

    % initialise parameters
    % y_scale*** will keep the intermediate steps, no need to keep (r x c x numScales) as even numbers
    % are not used
    y_scaleMax(rows,cols,ceil(numScales/2)) = 0;
    y_scaleMin(rows,cols,ceil(numScales/2)) = 0;
    dataOut(rows,cols,levs)                 = 0;        %#ok<NASGU>
    sizeFR_S                                = 3;        % size of a Gaussian filter to remove noise
    sizeFC_S                                = sizeFR_S;
    filtG_small                             = gaussF(sizeFR_S,sizeFC_S,1); %define the filter

    %---- Low pass filter to reduce the effect of noise
    y1                                      = conv2(padData(dataIn,ceil(sizeFR_S/2)),filtG_small);
    y_LPF                                   = y1(sizeFR_S+1:end-sizeFR_S,sizeFC_S+1:end-sizeFC_S);

    %clear y_scaleM*;

    %---- adjust a surface to the maxima/minima of the data

    for cStep=1:2:numScales
        %disp(cStep)
        %each scale will find the average of the opposite neighbours of
        %a pixel at different degrees of separation
        cStep2                              = ceil(cStep/2);
        y_scaleMax(:,:,cStep2)              = y_LPF;
        y_scaleMin(:,:,cStep2)              = y_LPF;

        %diagonal neighbours of an 8-connectivity
        tempNW                              = y_LPF (1:rows-2*cStep,1:cols-2*cStep);
        tempSW                              = y_LPF (1+2*cStep:rows,1:cols-2*cStep);
        tempSE                              = y_LPF (1+2*cStep:rows,1+2*cStep:cols);
        tempNE                              = y_LPF (1:rows-2*cStep,1+2*cStep:cols);
        %immediate neighbours of a 4-connectivity
        tempN                               = y_LPF (1:rows-2*cStep,1+cStep:cols-cStep);
        tempW                               = y_LPF (1+cStep:rows-cStep,1:cols-2*cStep);
        tempS                               = y_LPF (1+2*cStep:rows,1+cStep:cols-cStep);
        tempE                               = y_LPF (1+cStep:rows-cStep,1+2*cStep:cols);
        clear tempAv*;

        %find averages of opposites and store vertically in a 3D matrix
        tempAv(:,:,1)                       = (tempNE+ tempSW)/2;
        tempAv(:,:,2)                       = (tempNW+ tempSE)/2;
        tempAv(:,:,3)                       = (tempN + tempS) /2;
        tempAv(:,:,4)                       = (tempW + tempE) /2;
        %find minimum and maximum
        tempAvMax                           = max (tempAv,[],3);
        tempAvMin                           = min (tempAv,[],3);
        %at each scale compare the averages with the actual value, keep the min/max
        y_scaleMax(1+cStep:rows-cStep,1+cStep:cols-cStep,cStep2)=max(y_scaleMax(1+cStep:rows-cStep,1+cStep:cols-cStep),tempAvMax);
        y_scaleMin(1+cStep:rows-cStep,1+cStep:cols-cStep,cStep2)=min(y_scaleMin(1+cStep:rows-cStep,1+cStep:cols-cStep),tempAvMin);

        %---- find the derivatives of the max/min envelope at the current level (including all levels so
        %far)

        % find the absolute min/max at all scales

        yMin                                = min(y_scaleMin(:,:,1:1:cStep2),[],3);
        yMax                                = max(y_scaleMax(:,:,1:1:cStep2),[],3);
        %

        sizeFR_S                            = cStep;
        sizeFC_S                            = sizeFR_S;  
        filtG_small                         = gaussF(sizeFR_S,sizeFC_S,1);

        %-----------------------------This is the most time consuming step ------------------------------
        %------------ since this is only a criterion to stop the process before it reaches the ----------
        %------------ last step, subsample to reduce complexity -----------------------------------------
        %---- Low pass filter to adjust the effects of high/low points
        
        yMin0                               = padData(yMin(1:2:end,1:2:end),ceil(sizeFR_S/2));
        yMin1                               = conv2(yMin0,filtG_small);
        yMin2                               = yMin1(sizeFR_S+1:end-sizeFR_S,sizeFC_S+1:end-sizeFC_S);
        yMax0                               = padData(yMax(1:2:end,1:2:end),ceil(sizeFR_S/2));
        yMax1                               = conv2(yMax0,filtG_small);
        yMax2                               = yMax1(sizeFR_S+1:end-sizeFR_S,sizeFC_S+1:end-sizeFC_S);
        
        


        %---- find the derivatives of the max/min envelope
        y_rderiv_Max                        = diff(yMax2,1,1);
        y_cderiv_Max                        = diff(yMax2,1,2);
        y_rderiv_Min                        = diff(yMin2,1,1);
        y_cderiv_Min                        = diff(yMin2,1,2);
        %
        %magnitude of the gradient
        y_magGrad_Max                       = sqrt(y_rderiv_Max(:,2:end).^2+y_cderiv_Max(2:end,:).^2);
        y_magGrad_Min                       = sqrt(y_rderiv_Min(:,2:end).^2+y_cderiv_Min(2:end,:).^2);
        %

        tot_grad_max1(cStep)                = sum(sum(y_magGrad_Max)); %#ok<AGROW>
        tot_grad_min1(cStep)                = sum(sum(y_magGrad_Min)); %#ok<AGROW>

        %
        if (cStep>1)

            diffGradMax                     =  abs((tot_grad_max1(end)-tot_grad_max1(end-2))/tot_grad_max1(end-2));
            diffGradMin                     =  abs((tot_grad_min1(end)-tot_grad_min1(end-2))/tot_grad_min1(end-2));

            %disp([cStep diffGradMin diffGradMax ] )
            if (diffGradMax<stopCriterion)||(diffGradMin<stopCriterion)
                break;
            end
        end
        %
    end

    %
    tot_grad_max                            = tot_grad_max1(end);
    tot_grad_min                            = tot_grad_min1(end);
    %%
    %compare the gradients to decide which surface to keep (smallest) but first re calculate the smooth


    yMin0                               = padData(yMin(1:1:end,1:1:end),ceil(sizeFR_S/2));
    yMin1                               = conv2(yMin0,filtG_small);
    yMin2                               = yMin1(sizeFR_S+1:end-sizeFR_S,sizeFC_S+1:end-sizeFC_S);
    yMax0                               = padData(yMax(1:1:end,1:1:end),ceil(sizeFR_S/2));
    yMax1                               = conv2(yMax0,filtG_small);
    yMax2                               = yMax1(sizeFR_S+1:end-sizeFR_S,sizeFC_S+1:end-sizeFC_S);


    if abs(((tot_grad_max-tot_grad_min)/tot_grad_max))<0.05
        yProm                               = 0.5*yMin2+0.5*yMax2;
        % dataOut(:,:,currLev)=dataIn(:,:,currLev)-(yProm)+mean(yProm(:));
        errSurface                          =  (yProm)-mean(yProm(:));
        dataOut                             = dataIn-errSurface;
    else
        if tot_grad_max>tot_grad_min
            % dataOut(:,:,currLev)=dataIn(:,:,currLev)-(yMin2)+mean(yMin2(:));
            %dataOut=dataIn-(yMin2)+mean(yMin2(:));
            errSurface                      =  (yMin2)-mean(yMin2(:));
            dataOut                         = dataIn-errSurface;
        else
            % dataOut(:,:,currLev)=dataIn(:,:,currLev)-(yMax2)+mean(yMax2(:));
            errSurface                      =  (yMax2)-mean(yMax2(:));
            dataOut                         = dataIn-errSurface;
            %%dataOut=dataIn-(yMax2)+mean(yMax2(:));
        end
    end
    if nargout==4
        errSurfaceMax                       = (yMax2);
        errSurfaceMin                       = (yMin2);
    end
    if reconvertTo255 ==1
        dataOut(dataOut>255)                    = 255;
        dataOut(dataOut<0)                      = 0;
    end
    %dataOut                     = (dataOut-min(dataOut(:)));
    %dataOut                     = 255*(dataOut/max(dataOut(:)));
end


function [gauss]=gaussF(rowDim,colDim,levDim,rowSigma,colSigma,levSigma,rowMiu,colMiu,levMiu,rho)
%function [gauss]=gaussF(rowDim,colDim,levDim,rowSigma,colSigma,levSigma,rowMiu,colMiu,levMiu,rho)
%
%--------------------------------------------------------------------------
% gaussF  produces an N-dimensional gaussian function (N=1,2,3)
%
%       INPUT
%         rowDim,colDim,levDim:       dimensions x,y,z
%         rowSigma,colSigma,levSigma: standard deviation values x,y,z >0
%         rowMiu,colMiu,levMiu:       mean values.
%         rho:                        (-1 1) oblique distributions angle 
%                                       control.
%
%       OUTPUT
%         gauss:                      n-dimensional gaussian function           
%          
%--------------------------------------------------------------------------
%
%     Copyright (C) 2012  Constantino Carlos Reyes-Aldasoro
%
%     This file is part of the PhagoSight package.
%
%     The PhagoSight package is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, version 3 of the License.
%
%     The PhagoSight package is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
%
%     You should have received a copy of the GNU General Public License
%     along with the PhagoSight package.  If not, see <http://www.gnu.org/licenses/>.
%
%--------------------------------------------------------------------------
%
% This m-file is part of the PhagoSight package used to analyse fluorescent phagocytes
% as observed through confocal or multiphoton microscopes.  For a comprehensive 
% user manual, please visit:
%
%           http://www.phagosight.org.uk
%
% Please feel welcome to use, adapt or modify the files. If you can improve
% the performance of any other algorithm please contact us so that we can
% update the package accordingly.
%
%--------------------------------------------------------------------------
%
% The authors shall not be liable for any errors or responsibility for the 
% accuracy, completeness, or usefulness of any information, or method in the content, or for any 
% actions taken in reliance thereon.
%
%--------------------------------------------------------------------------



%------ no input data is received, error -------------------------
if nargin<1 help gaussF;  gauss=[]; return; end;

%-----------------------------------------------------------------
%------ cases of input:                  -------------------------
%-----------------------------------------------------------------
% 1 only R,C,L, dimensions are specified,
%        then, set sigma so that borders are 1% of central value
%        sigma is set after miu is calculated
% 2 R,C,L dimensions are specified AND
%        sigmas are provided, then
%        set values of miu
% 3 all input arguments are provided
% 4 Rho is provided, in all the previous cases rho=0

if nargin<10 rho=0; end;


%-----------------------------------------------------------------
%------ Determine the dimensios of the gaussian function ---------
%------ dimensions can be input vectors as in (size(a)) ----------
%-----------------------------------------------------------------
if nargin==1
    [wRow,wCol,wLev]=size(rowDim);
    if wCol==3       %------ 3 D
        levDim=rowDim(3);      colDim=rowDim(2);      rowDim=rowDim(1);
    elseif wCol==2   %------ 2 D  set levels =1
        colDim=rowDim(2);      rowDim=rowDim(1);      levDim=1;
    elseif wCol==1   %------ 1 D is required, set others =1
        colDim=1;      levDim=1;
    end;
elseif nargin==2
    levDim=1;
end
%-----------------------------------------------------------------
%----- x, y, z dimensions of the filter --------------------------
%-----------------------------------------------------------------
filter.x=1:ceil(rowDim);
filter.y=1:ceil(colDim);
filter.z=1:ceil(levDim);
filter.data=zeros(ceil(rowDim),ceil(colDim),ceil(levDim));
[rr,cc,dd]=meshgrid(filter.x, filter.y ,filter.z);
%-----------------------------------------------------------------
%----- Determine mius and sigmas in case not provided ------------
%-----------------------------------------------------------------
if nargin<=6  %------ mean values are not provided
    rowMiu=sum(filter.x)/length(filter.x);
    colMiu=sum(filter.y)/length(filter.y);
    levMiu=sum(filter.z)/length(filter.z);
end

sigmVal=1.1774;

if nargin<=3    %------ sigma values are not provided
    rowSigma=(rowMiu-1)/sigmVal;
    colSigma=(colMiu-1)/sigmVal;
    levSigma=(levMiu-1)/sigmVal;
end;
%-----------------------------------------------------------------
%------ set value for 0.1% --> sqrt(2*log(0.001)) = 3.7169  ------
%------ set value for 1% --> sqrt(2*log(0.01)) = 3.0349     ------
%------ set value for 10% --> sqrt(2*log(0.1)) = 2.1460     ------
%------ set value for 50% --> sqrt(2*log(0.5)) = 1.1774     ------

%-----------------------------------------------------------------
%------ sigma must be greater than zero --------------------------
rowSigma=max(rowSigma,0.000001);
colSigma=max(colSigma,0.000001);
levSigma=max(levSigma,0.000001);

if prod(size(rho))~=1
    %rho is the covariance matrix
    if size(rho,1)==2
        invSigma=inv(rho);
        Srr=invSigma(1,1);Scc=invSigma(2,2);Src=2*invSigma(2,1);
        Srd=0;Scd=0;Sdd =0;
    else
        invSigma=inv(rho);
        Srr=invSigma(1,1);Scc=invSigma(2,2);Src=2*invSigma(2,1);
        Srd=2*invSigma(1,3);Scd=2*invSigma(2,3);Sdd=invSigma(3,3);
    end
    exp_r= (1/rowSigma/rowSigma)*(rr-rowMiu).^2 ;
    exp_c=(1/colSigma/colSigma)*(cc-colMiu).^2 ;
    exp_d=(1/levSigma/levSigma)*(dd-levMiu).^2;
    exp_rc=(1/rowSigma/colSigma)*(rr-rowMiu).*(cc-colMiu);
    exp_rd=(1/rowSigma/levSigma)*(rr-rowMiu).*(dd-levMiu);
    exp_cd=(1/levSigma/colSigma)*(dd-levMiu).*(cc-colMiu);
    gauss=exp(-(Srr * exp_r + Scc * exp_c  + Sdd * exp_d + Src * exp_rc + Srd * exp_rd + Scd * exp_cd ));
else 




    rho=min(rho,0.999999);
    rho=max(rho,-0.999999);

    %-----------------------------------------------------------------
    %------ Calculate exponential functions in each dimension --------
    filter.x2=(1/(sqrt(2*pi)*rowSigma))*exp(-((filter.x-rowMiu).^2)/2/rowSigma/rowSigma);
    filter.y2=(1/(sqrt(2*pi)*colSigma))*exp(-((filter.y-colMiu).^2)/2/colSigma/colSigma);
    filter.z2=(1/(sqrt(2*pi)*levSigma))*exp(-((filter.z-levMiu).^2)/2/levSigma/levSigma);

    %------ ? ? ? ? The individual functions should add to 1 ? ? ? ---
    filter.x2=filter.x2/sum(filter.x2);
    filter.y2=filter.y2/sum(filter.y2);
    filter.z2=filter.z2/sum(filter.z2);
    %-----------------------------------------------------------------
    rhoExponent=(-(rho*(filter.x-rowMiu)'*(filter.y-colMiu))/rowSigma/colSigma);
    filter.rho=(1/sqrt(1-rho^2))*exp(rhoExponent);
    %-----------------------------------------------------------------
    %------ Get the 2D function  (if needed)--------------------------
    if (colDim>1 & rowDim>1)
        twoDFilter=(filter.x2'*filter.y2).*filter.rho;
        %------ Get the 3D function  (if needed)----------------------
        if ceil(levDim)>1
            for ii=1:ceil(levDim);
                threeDFilter(:,:,ii)=twoDFilter.*filter.z2(ii);
            end;
            gauss=threeDFilter;
        else
            gauss=twoDFilter;
        end;
    else    %------This covers the 1D cases both row and column ------
        if length(filter.x2)>length(filter.y2)
            gauss=filter.x2;
        else
            gauss=filter.y2;
        end
    end;

    %------ remove NaN in case there are any
    gauss(isnan(gauss))=0;
end



function dataPadded=padData (qtdata,numPadPixels,dimsToPad,padWith)
%function dataPadded=padData (qtdata,numPadPixels,dimsToPad,padWith)
%-----------------------------------------------------------------
%------  Author :   Constantino Carlos Reyes-Aldasoro-------------
%------             PHD     the University of Warwick-------------
%------  Supervisor :   Abhir Bhalerao    ------------------------
%------  5 March 2002    -----------------------------------------
%-----------------------------------------------------------------
%------ input  :  The data to be padded, default is to pad   -----
%------           with SAME values on the edges              -----
%------           but it can be changed with padWith to zero -----
%------           numPadPixels determine padding area        -----
%------ output :  the padded data
%-----------------------------------------------------------------
%----------------------------------------------------
%------ For a description and explanation please refer to:
%------ http://www.dcs.warwick.ac.uk/~creyes/m-vts --
%----------------------------------------------------

if ~exist('padWith') padWith=1; end
%-----dimensions will determine if it is a 1D, 2D, 3D, 4D ... padding
if (~exist('dimsToPad'))|(isempty(dimsToPad))
    [rows,cols,levs,numFeats]=size(qtdata);
else
    dimsToPad=[dimsToPad ones(1,4)]; %pad with ones in case it is too short
    rows=dimsToPad(1);
    cols=dimsToPad(2);
    levs=dimsToPad(3);
    numFeats=dimsToPad(4);
end

if rows>1    %---- first pad the rows
    qtdata=[padWith*repmat(qtdata(1,:,:,:),[numPadPixels 1 1]); qtdata;  padWith*repmat(qtdata(end,:,:,:),[numPadPixels 1 1])];
end
if cols>1    %---- then pad the cols
    qtdata=[padWith*repmat(qtdata(:,1,:,:),[1 numPadPixels 1]) qtdata  padWith*repmat(qtdata(:,end,:,:),[1 numPadPixels 1])];
end
dataPadded=qtdata;
if levs>1
    [rows,cols,levs,numFeats]=size(qtdata);
        qtdata3(:,:,numPadPixels+1:numPadPixels+levs,:)=qtdata;
        %qtdata3(:,:,numPadPixels+1:levs-numPadPixels,:)=qtdata;
        qtdata3(:,:,1:numPadPixels,:)=padWith*repmat(qtdata(:,:,1,:),[1 1 numPadPixels]);
        qtdata3(:,:,numPadPixels+levs+1:2*numPadPixels+levs,:)=padWith*repmat(qtdata(:,:,end,:),[1 1 numPadPixels]);
        %qtdata3(:,:,levs+1-numPadPixels:levs,:)=repmat(qtdata(:,:,end,:),[1 1 numPadPixels]);
        dataPadded=qtdata3;
end   
