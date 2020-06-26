from sklearn.metrics import confusion_matrix
import torch
from tqdm.notebook import tqdm
def num_correct(prediction,labels):
    correct=0
    for i,(pred_label,label) in enumerate(zip(prediction,labels)):
        if (pred_label.item()==label.item()):
            correct +=1
    return correct



def face_train(master_path,num_epochs,name,model,train_dataloader,valid_dataloader,optimizer,criterion,device):
    f = open(master_path +"/"+name+".txt",'a')
    #Triaining
    train_loss=[]
    valid_accuracy=[]
    model.train()
    for epochs in range(0,num_epochs):        
        print("Epoch: ", epochs+1,"\n")
        model.eval()   
        correct=0
        total_samples=0
        avg_vloss=0
        first=True
        
        print("Validation \n")
        for i_batch, (_,face_batch,_,label) in enumerate(valid_dataloader):
            batch_size=face_batch.size(0)
            face_batch=face_batch.float().to(device)
            output=model(None,face_batch)
            loss=criterion(output,label.to(device))
            avg_vloss+=loss.item()
            predicted = torch.max(output, 1)
            prediction=predicted.indices.detach().cpu()
            correct +=num_correct(prediction,label)
            total_samples+=batch_size
            true_label=label.detach().cpu()
              
              
            if first:
                first=False
                conf_mat=confusion_matrix( true_label,prediction,labels=[0,1,2])
            else:
                conf_mat+=confusion_matrix(true_label,prediction,labels=[0,1,2])
        
        avg_vloss=avg_vloss/len(valid_dataloader)
        print("Validation Loss: ", avg_vloss)
        avg_vaccuracy=correct/(total_samples)
        print("Validation Accuracy: ", avg_vaccuracy)
        print('Confusion Matrix: \n',conf_mat)          

        model.train()
        correct=0
        total_samples=0
        avg_tloss=0
        first=True
        for i_batch, (_,face_batch,_,label) in tqdm(enumerate(train_dataloader)):
         
            batch_size=face_batch.size(0)
            optimizer.zero_grad()
            face_batch=face_batch.float().to(device)
           
            output=model(None,face_batch)
            loss=criterion(output,label.to(device))
            loss.backward()
            predicted = torch.max(output, 1)
            prediction=predicted.indices.detach().cpu()
            correct +=num_correct(prediction,label)
            total_samples+=batch_size
            accuracy=correct/(total_samples)
            optimizer.step()
            true_label=label.detach().cpu()
            avg_tloss+=loss.item()
            if first:
                first=False
                conf_mat=confusion_matrix( true_label,prediction,labels=[0,1,2])
            else:
                conf_mat+=confusion_matrix(true_label,prediction,labels=[0,1,2])
            if (i_batch+1)%40==0:
               # print(label)
                print("Batch: ",i_batch+1,"/",len(train_dataloader))
                print("Batch Recognition loss: ", loss.item())

        avg_tloss=avg_tloss/len(train_dataloader)
        avg_taccuracy=correct/total_samples
        print("Average_Loss: ",avg_tloss)
        print("Average_Accuracy: ",avg_taccuracy)
        print('Confusion Matrix: \n',conf_mat)
        if epochs>0:
            torch.save(model.state_dict(),master_path+"/"+name+"_"+str(epochs+1)+".pth")
        
        data = " %f,%f,%f,%f \n" % (avg_tloss,avg_taccuracy,avg_vloss,avg_vaccuracy)
        f.write(data)



    f.close()
              
def frame_train(master_path,num_epochs,name,model,train_dataloader,valid_dataloader,optimizer,criterion,device):
    f = open(master_path +"/"+name+".txt",'a')
    #Triaining
    train_loss=[]
    valid_accuracy=[]
    model.train()
    for epochs in range(0,num_epochs):        
        print("Epoch: ", epochs+1,"\n")
        model.eval()   
        correct=0
        total_samples=0
        avg_vloss=0
        first=True
        
        print("Validation \n")
        for i_batch, (frame_batch,_,_,label) in enumerate(valid_dataloader):
            batch_size=frame_batch.size(0)       
            frame_batch=frame_batch.float().to(device)
            output=model(frame_batch,None)
            loss=criterion(output,label.to(device))
            avg_vloss+=loss.item()
            predicted = torch.max(output, 1)
            prediction=predicted.indices.detach().cpu()
            correct +=num_correct(prediction,label)
            total_samples+=batch_size
            true_label=label.detach().cpu()
              
              
            if first:
                first=False
                conf_mat=confusion_matrix( true_label,prediction,labels=[0,1,2])
            else:
                conf_mat+=confusion_matrix(true_label,prediction,labels=[0,1,2])
        
        avg_vloss=avg_vloss/len(valid_dataloader)
        print("Validation Loss: ", avg_vloss)
        avg_vaccuracy=correct/(total_samples)
        print("Validation Accuracy: ", avg_vaccuracy)
        print('Confusion Matrix: \n',conf_mat)          

        model.train()
        correct=0
        total_samples=0
        avg_tloss=0
        first=True
        for i_batch, (frame_batch,_,_,label) in tqdm(enumerate(train_dataloader)):
         
            batch_size=frame_batch.size(0)
            optimizer.zero_grad()
            batch_size=frame_batch.size(0)       
            frame_batch=frame_batch.float().to(device)
            output=model(frame_batch,None)
            loss=criterion(output,label.to(device))
            loss.backward()
            predicted = torch.max(output, 1)
            prediction=predicted.indices.detach().cpu()
            correct +=num_correct(prediction,label)
            total_samples+=batch_size
            accuracy=correct/(total_samples)
            optimizer.step()
            true_label=label.detach().cpu()
            avg_tloss+=loss.item()
            if first:
                first=False
                conf_mat=confusion_matrix( true_label,prediction,labels=[0,1,2])
            else:
                conf_mat+=confusion_matrix(true_label,prediction,labels=[0,1,2])
            if (i_batch+1)%40==0:
               # print(label)
                print("Batch: ",i_batch+1,"/",len(train_dataloader))
                print("Batch Recognition loss: ", loss.item())

        avg_tloss=avg_tloss/len(train_dataloader)
        avg_taccuracy=correct/total_samples
        print("Average_Loss: ",avg_tloss)
        print("Average_Accuracy: ",avg_taccuracy)
        print('Confusion Matrix: \n',conf_mat)
        if epochs>=0:
            torch.save(model.state_dict(),master_path+"/"+name+"_"+str(epochs+1)+".pth")
        
        data = " %f,%f,%f,%f \n" % (avg_tloss,avg_taccuracy,avg_vloss,avg_vaccuracy)
        f.write(data)



    f.close()


def image_train(master_path,num_epochs,name,model,train_dataloader,valid_dataloader,optimizer,criterion,device):
    
    #Triaining
    train_loss=[]
    valid_accuracy=[]
    model.train()
    for epochs in range(0,num_epochs): 
        f = open(master_path +"/"+name+".txt",'a')
        print("Epoch: ", epochs+1,"\n")
        model.eval()   
        correct=0
        total_samples=0
        avg_vloss=0
        first=True
        
        print("Validation \n")
        for i_batch, (frame_batch,face_batch,_,label) in enumerate(valid_dataloader):
            batch_size=frame_batch.size(0)
            face_batch=face_batch.float().to(device)
            frame_batch=frame_batch.float().to(device)
            output=model(frame_batch,face_batch)
            loss=criterion(output,label.to(device))
            avg_vloss+=loss.item()
            predicted = torch.max(output, 1)
            prediction=predicted.indices.detach().cpu()
            correct +=num_correct(prediction,label)
            total_samples+=batch_size
            true_label=label.detach().cpu()
              
              
            if first:
                first=False
                conf_mat=confusion_matrix( true_label,prediction,labels=[0,1,2])
            else:
                conf_mat+=confusion_matrix(true_label,prediction,labels=[0,1,2])
        
        avg_vloss=avg_vloss/len(valid_dataloader)
        print("Validation Loss: ", avg_vloss)
        avg_vaccuracy=correct/(total_samples)
        print("Validation Accuracy: ", avg_vaccuracy)
        print('Confusion Matrix: \n',conf_mat)          

        model.train()
        correct=0
        total_samples=0
        avg_tloss=0
        first=True
        for i_batch, (frame_batch,face_batch,_,label) in enumerate(train_dataloader):
            optimizer.zero_grad()
            batch_size=frame_batch.size(0)
            face_batch=face_batch.float().to(device)
            frame_batch=frame_batch.float().to(device)
            output=model(frame_batch,face_batch)
            loss=criterion(output,label.to(device))
            loss.backward()
            predicted = torch.max(output, 1)
            prediction=predicted.indices.detach().cpu()
            correct +=num_correct(prediction,label)
            total_samples+=batch_size
            accuracy=correct/(total_samples)
            optimizer.step()
            true_label=label.detach().cpu()
            avg_tloss+=loss.item()
            if first:
                first=False
                conf_mat=confusion_matrix( true_label,prediction,labels=[0,1,2])
            else:
                conf_mat+=confusion_matrix(true_label,prediction,labels=[0,1,2])
            if (i_batch+1)%40==0:
               # print(label)
                print("Batch: ",i_batch+1,"/",len(train_dataloader))
                print("Batch Recognition loss: ", loss.item())

        avg_tloss=avg_tloss/len(train_dataloader)
        avg_taccuracy=correct/total_samples
        print("Average_Loss: ",avg_tloss)
        print("Average_Accuracy: ",avg_taccuracy)
        print('Confusion Matrix: \n',conf_mat)
        if epochs>0:
            torch.save(model.state_dict(),master_path+"/"+name+"_"+str(epochs+1)+".pth")
        
        data = " %f,%f,%f,%f \n" % (avg_tloss,avg_taccuracy,avg_vloss,avg_vaccuracy)
        f.write(data)



    f.close()

              
def audio_train(master_path,num_epochs,name,model,train_dataloader,valid_dataloader,optimizer,criterion,device):
    f = open(master_path +"/"+name+".txt",'a') 
              #Triaining
    train_loss=[]
    valid_accuracy=[]
    model.train()
    for epochs in range(0,num_epochs):        
        print("Epoch: ", epochs+1,"\n")
        model.eval()   
        correct=0
        total_samples=0
        avg_vloss=0
        first=True
        
        print("Validation \n")
        for i_batch, (_,_,audio_feature,label) in enumerate(valid_dataloader):
            batch_size=audio_feature.size(0)
            audio_feature=audio_feature.float().to(device)
            output=model(audio_feature)
            loss=criterion(output,label.to(device))
            avg_vloss+=loss.item()
            
            predicted = torch.max(output, 1)
            prediction=predicted.indices.detach().cpu()
            correct +=num_correct(prediction,label)
            total_samples+=batch_size
            true_label=label.detach().cpu()
              
              
            if first:
                first=False
                conf_mat=confusion_matrix( true_label,prediction,labels=[0,1,2])
            else:
                conf_mat+=confusion_matrix(true_label,prediction,labels=[0,1,2])
        
        avg_vloss=avg_vloss/len(valid_dataloader)
        print("Validation Loss: ", avg_vloss)
        avg_vaccuracy=correct/(total_samples)
        print("Validation Accuracy: ", avg_vaccuracy)
        print('Confusion Matrix: \n',conf_mat)          

        model.train()
        correct=0
        total_samples=0
        avg_tloss=0
        first=True
        for i_batch, (_,_,audio_feature,label) in tqdm(enumerate(train_dataloader)):
         
            batch_size=audio_feature.size(0)
            optimizer.zero_grad()
            audio_feature=audio_feature.float().to(device)
            output=model(audio_feature)
            loss=criterion(output,label.to(device))
            loss.backward()
            predicted = torch.max(output, 1)
            prediction=predicted.indices.detach().cpu()
            correct +=num_correct(prediction,label)
            total_samples+=batch_size
            accuracy=correct/(total_samples)
            optimizer.step()
            true_label=label.detach().cpu()
            avg_tloss+=loss.item()
            if first:
                first=False
                conf_mat=confusion_matrix( true_label,prediction,labels=[0,1,2])
            else:
                conf_mat+=confusion_matrix(true_label,prediction,labels=[0,1,2])
            if (i_batch+1)%40==0:
               # print(label)
                print("Batch: ",i_batch+1,"/",len(train_dataloader))
                print("Batch Recognition loss: ", loss.item())

        avg_tloss=avg_tloss/len(train_dataloader)
        avg_taccuracy=correct/total_samples
        print("Average_Loss: ",avg_tloss)
        print("Average_Accuracy: ",avg_taccuracy)
        print('Confusion Matrix: \n',conf_mat)
        if epochs+1>0:
            torch.save(model.state_dict(),master_path+"/"+name+"_"+str(epochs+1)+".pth")
        
        data = " %f,%f,%f,%f \n" % (avg_tloss,avg_taccuracy,avg_vloss,avg_vaccuracy)
        f.write(data)



    f.close()

def full_train(master_path,num_epochs,name,model,train_dataloader,valid_dataloader,optimizer,criterion,device):
    f = open(master_path +"/"+name+".txt",'a') 
              #Triaining
    train_loss=[]
    valid_accuracy=[]
    model.train()
    for epochs in range(0,num_epochs):        
        print("Epoch: ", epochs+1,"\n")
        model.eval()   
        correct=0
        total_samples=0
        avg_vloss=0
        first=True
        
        print("Validation \n")
        for i_batch, (_,frame_batch,face_batch,audio_feature,label) in enumerate(valid_dataloader):
            batch_size=audio_feature.size(0)
            face_batch=face_batch.float().to(device)
            frame_batch=frame_batch.float().to(device)
            audio_feature=audio_feature.float().to(device)
            output=model(frame_batch,face_batch,audio_feature)
            loss=criterion(output,label.to(device))
            avg_vloss+=loss.item()
            predicted = torch.max(output, 1)
            prediction=predicted.indices.detach().cpu()
            correct +=num_correct(prediction,label)
            total_samples+=batch_size
            true_label=label.detach().cpu()
              
              
            if first:
                first=False
                conf_mat=confusion_matrix( true_label,prediction,labels=[0,1,2])
            else:
                conf_mat+=confusion_matrix(true_label,prediction,labels=[0,1,2])
        
        avg_vloss=avg_vloss/len(valid_dataloader)
        print("Validation Loss: ", avg_vloss)
        avg_vaccuracy=correct/(total_samples)
        print("Validation Accuracy: ", avg_vaccuracy)
        print('Confusion Matrix: \n',conf_mat)          
        print("Train \n")
        model.train()
        correct=0
        total_samples=0
        avg_tloss=0
        first=True
        for i_batch, (_,frame_batch,face_batch,audio_feature,label) in tqdm(enumerate(train_dataloader)):
         
            batch_size=audio_feature.size(0)
            optimizer.zero_grad()
            face_batch=face_batch.float().to(device)
            frame_batch=frame_batch.float().to(device)
            audio_feature=audio_feature.float().to(device)
            output=model(frame_batch,face_batch,audio_feature)
            loss=criterion(output,label.to(device))
            loss.backward()
            predicted = torch.max(output, 1)
            prediction=predicted.indices.detach().cpu()
            correct +=num_correct(prediction,label)
            total_samples+=batch_size
            accuracy=correct/(total_samples)
            optimizer.step()
            true_label=label.detach().cpu()
            avg_tloss+=loss.item()
            if first:
                first=False
                conf_mat=confusion_matrix( true_label,prediction,labels=[0,1,2])
            else:
                conf_mat+=confusion_matrix(true_label,prediction,labels=[0,1,2])
            if (i_batch+1)%40==0:
               # print(label)
                print("Batch: ",i_batch+1,"/",len(train_dataloader))
                print("Batch Recognition loss: ", loss.item())

        avg_tloss=avg_tloss/len(train_dataloader)
        avg_taccuracy=correct/total_samples
        print("Average_Loss: ",avg_tloss)
        print("Average_Accuracy: ",avg_taccuracy)
        print('Confusion Matrix: \n',conf_mat)
        if epochs>0:
            torch.save(model.state_dict(),master_path+"/"+name+"_"+str(epochs+1)+".pth")
        
        data = " %f,%f,%f,%f \n" % (avg_tloss,avg_taccuracy,avg_vloss,avg_vaccuracy)
        f.write(data)



    f.close()

def full_train_global(master_path,num_epochs,name,model,train_dataloader,valid_dataloader,optimizer,criterion,device):
    f = open(master_path +"/"+name+".txt",'a') 
              #Triaining
    train_loss=[]
    valid_accuracy=[]
    model.train()
    for epochs in range(0,num_epochs):        
        print("Epoch: ", epochs+1,"\n")
        model.eval()   
        correct=0
        total_samples=0
        avg_vloss=0
        first=True
        
        print("Validation \n")
        for i_batch, (_,frame_batch,audio_feature,label) in enumerate(valid_dataloader):
            batch_size=audio_feature.size(0)
            frame_batch=frame_batch.float().to(device)
            audio_feature=audio_feature.float().to(device)
            output=model(frame_batch,audio_feature)
            loss=criterion(output,label.to(device))
            avg_vloss+=loss.item()
            predicted = torch.max(output, 1)
            prediction=predicted.indices.detach().cpu()
            correct +=num_correct(prediction,label)
            total_samples+=batch_size
            true_label=label.detach().cpu()
              
              
            if first:
                first=False
                conf_mat=confusion_matrix( true_label,prediction,labels=[0,1,2])
            else:
                conf_mat+=confusion_matrix(true_label,prediction,labels=[0,1,2])
        
        avg_vloss=avg_vloss/len(valid_dataloader)
        print("Validation Loss: ", avg_vloss)
        avg_vaccuracy=correct/(total_samples)
        print("Validation Accuracy: ", avg_vaccuracy)
        print('Confusion Matrix: \n',conf_mat)          
        print("Train \n")
        model.train()
        correct=0
        total_samples=0
        avg_tloss=0
        first=True
        for i_batch, (_,frame_batch,audio_feature,label) in tqdm(enumerate(train_dataloader)):
         
            batch_size=audio_feature.size(0)
            optimizer.zero_grad()
            frame_batch=frame_batch.float().to(device)
            audio_feature=audio_feature.float().to(device)
            output=model(frame_batch,audio_feature)
            loss=criterion(output,label.to(device))
            loss.backward()
            predicted = torch.max(output, 1)
            prediction=predicted.indices.detach().cpu()
            correct +=num_correct(prediction,label)
            total_samples+=batch_size
            accuracy=correct/(total_samples)
            optimizer.step()
            true_label=label.detach().cpu()
            avg_tloss+=loss.item()
            if first:
                first=False
                conf_mat=confusion_matrix( true_label,prediction,labels=[0,1,2])
            else:
                conf_mat+=confusion_matrix(true_label,prediction,labels=[0,1,2])
            if (i_batch+1)%40==0:
               # print(label)
                print("Batch: ",i_batch+1,"/",len(train_dataloader))
                print("Batch Recognition loss: ", loss.item())

        avg_tloss=avg_tloss/len(train_dataloader)
        avg_taccuracy=correct/total_samples
        print("Average_Loss: ",avg_tloss)
        print("Average_Accuracy: ",avg_taccuracy)
        print('Confusion Matrix: \n',conf_mat)
        if epochs>0:
            torch.save(model.state_dict(),master_path+"/"+name+"_"+str(epochs+1)+".pth")
        
        data = " %f,%f,%f,%f \n" % (avg_tloss,avg_taccuracy,avg_vloss,avg_vaccuracy)
        f.write(data)



    f.close()