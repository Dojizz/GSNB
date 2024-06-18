#### Compilation
Currently I only plan running the project on Windows. For Linux, refer to the official doc.

```sh
## use x64 native tools prompt to run the command
cmake -Bbuild .
cmake --build build --target install --config RelWithDebInfo
```

#### Running
Open `build/sibr_projects.sln` to run the project, set projects/gaussian/apps/SIBR_gaussianViewer_app as the start project. Configure the command to open the trained model. The model(trained gaussian file) should be placed in the `models` folder, so that they won't be transfered to the remote repo.