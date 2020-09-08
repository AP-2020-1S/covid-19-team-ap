call activate 
call python.exe casos.py 
call python.exe "Actualizar modelos.py"
call git pull
call git add *
call git commit -m "actualizacion diaria de los datos y modelos"
call git push origin master
call heroku container:login 
call heroku container:push web -a teamap-unal 
call heroku container:release web -a teamap-unal 
