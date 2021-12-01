

function con = mnist_import( path )

      str = fileread(path);
      data = jsondecode(str);
   
      
      dim = data.( 'mnist' ).size;
      con = struct('x', zeros( dim(2), dim(1)),...
                   'y', zeros( dim(3), dim(1)) );
             
      for i = 1:dim(1)
            datapoint =  data.( 'mnist' ).data(i);
            
            con.x(:,i) = [ datapoint.x ]';
            con.y(:,i) = [ datapoint.y ]';
      end
      
end