

   computer_model = computer;
   matlabversion = sscanf(version,'%f');
   matlabversion = matlabversion(1);
%%
   if strcmp(computer_model,'PCWIN')
      str1 = [matlabroot,'\extern\lib\win32\lcc\libmwlapack.lib  '];
      str2 = [matlabroot,'\extern\lib\win32\lcc\libmwblas.lib  '];
      libstr = [str1,str2];     
   elseif strcmp(computer_model,'PCWIN64')
      str1 = [matlabroot,'\extern\lib\win64\lcc\libmwlapack.lib  '];
      str2 = [matlabroot,'\extern\lib\win64\lcc\libmwblas.lib  '];
      libstr = [str1,str2];  
   else
      libstr = ' -lmwlapack -lmwblas  '; 
   end
   mexcmd = 'mex -O  -largeArrayDims  -output ';    

   eval([mexcmd, 'mexeig mexeig.c',libstr]);  
   eval([mexcmd, 'mexsvd mexsvd.c',libstr]);  

   eval([mexcmd, 'reorth mexreorth.c',libstr]);
