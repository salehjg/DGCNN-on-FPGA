#include <iostream>
#include "FakeModelArchTop.h"

using namespace std;

int main(){

    {
        cout << "Memory Bank Optimizer..." << endl;
        ModelArchTop01 model(0, 5, 1024, 20);
        WorkScheduler scheduler;
        model.Execute(scheduler);
        int dataMoverLaunches = model.GetDataMoverLaunches();
        string objectiveStr="";
        for(string &tag:model.platformSelector->objective){
            objectiveStr += tag + " + ";
        }
        cout << "DataMoverLaunches: " << dataMoverLaunches << endl;
        cout << "Objective to minimize:\n"<<objectiveStr<<endl;
    }




}