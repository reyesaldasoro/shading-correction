function [expData]=expandu(data,numExpansions)
% function [expData]=expandu(data)
%----------------------------------------------------------------
%------ EXPANDU  function to expand data in uniform levels  -----
%------          receives an image and expands it in quadtree ---
%------          expantion is ALWAYS in factor of 2 -------------
%----------------------------------------------------------------
%----------------------------------------------------
%------  Author :   Constantino Carlos Reyes-Aldasoro
%------             PHD     the University of Warwick
%------  Supervisor :   Abhir Bhalerao    -----------
%------             2001 ----------------------------
%----------------------------------------------------
%----------------------------------------------------
%------ For a description and explanation please refer to:
%------ http://www.dcs.warwick.ac.uk/~creyes/m-vts --
%----------------------------------------------------

%------ no input data is received, error -------------------------
if nargin<1
    help expandu;
    expData=[];
    return;
end;
if isempty(data)
    expData=[];
    return;
end;

if ~exist('numExpansions')  numExpansions=1; end

if numExpansions>0
    if numExpansions>1
        data=expandu(data,numExpansions-1);    
    end
    [rows,cols,levels]=size(data);
    
    %-------------------general revision about the sizes ---------------------
    
    % %------ !!!!restriction!!! the data sizes a power of 2
    % if ((log2(rows)~=floor(log2(rows)))|(log2(cols)~=floor(log2(cols))))
    %     help expandu;
    %     warning('Data dimensions must be a power of 2');
    %     expData=[];
    %     
    %   
    %     
    %     return;
    % end
%    if rows~=cols
        if levels==1 expData=zeros(rows*2,cols*2); else  expData=zeros(rows*2,cols*2,levels*2);   end
        expData(1:2:end,1:2:end,1:levels)=data;
        expData(1:2:end,2:2:end,1:levels)=data;
        expData(2:2:end,1:2:end,1:levels)=data;
        
        expData(2:2:end,2:2:end,1:levels)=data;
        if levels>1 
            expData(:,:,1:2:end)=expData(:,:,1:levels); 
            expData(:,:,2:2:end)=expData(:,:,1:2:end); 
        end
%         return;
%     end
%     
%     
%     %------ matrix to multiply in order to reduce the data ---------
%     %------ basic matrix, Identity modified   ----------------------
%     if floor(rows/2)==(rows/2)
%         Imod=[1 1];                                                 %expands 1 to 2
%         Imodn=setImod(Imod,rows);
%         
%         %determine 2D and 3D Cases
%         if levels==1
%             expData=Imodn'*data*Imodn;
%         else
%             for k=1:levels
%                 expData(:,:,2*k)=Imodn'*data(:,:,k)*Imodn;
%                 expData(:,:,2*k-1)=expData(:,:,2*k);
%             end
%         end
%     else
%         expData=zeros(rows*2,cols*2,2*levels);
%         expData(1:2:end,1:2:end,1:2:end)=data;
%         expData(1:2:end,2:2:end,1:2:end)=data;
%         expData(2:2:end,1:2:end,1:2:end)=data;
%         expData(2:2:end,2:2:end,1:2:end)=data;
%         expData(:,:,2:2:end)=expData(:,:,1:2:end);
%     end
%     
% else
%     expData=data;
end