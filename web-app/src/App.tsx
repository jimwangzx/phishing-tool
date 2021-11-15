import './App.css';
import { Chart } from "react-google-charts";
import { TextBoxComponent } from '@syncfusion/ej2-react-inputs';
import axios from "axios"; 
import React, {  useEffect, useState } from "react";

const baseURL = "http://localhost:4433/phishing/text";

interface Idata {
  id: string;
  content: string;
  subject_line: string;
  email_address: string;
  textareaObj: any;
}


function App() {
	
	const [state, setState] = useState<Partial<Idata>>({ id: "1", content: "", subject_line: "", email_address: ""});
    const [phishing, setPhish] = useState(0); 

	useEffect(() => {
    	if (phishing === 0) {
      	console.log('Component loaded!')
    	} else {
      	console.log('Button A was clicked!');
    	}
  	}, [phishing]);	
 	
	const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
		console.log(state);
		const value: Idata = {
            "id": "1",
            "content": state.content!,
            "subject_line" : state.subject_line!,
            "email_address" : state.email_address!,
			"textareaObj": {} 
		}
		console.log(value)
    	axios
      		.post(baseURL,  value ).then(response => {
			console.log(response);
			setPhish(response.data);
		})
    }
     const onCreate = () => {
        state.textareaObj.addAttributes({rows: "1"});
        state.textareaObj.respectiveElement.style.height = "auto";
        state.textareaObj.respectiveElement.style.height = (state.textareaObj.respectiveElement .scrollHeight)+"px";
    }
    const onInput = () => {
        state.textareaObj.respectiveElement.style.height = "auto";
        state.textareaObj.respectiveElement.style.height = (state.textareaObj.respectiveElement .scrollHeight)+"px";
    }
    function setEmail(email_address: string) {
		setState(prevState => ({...prevState, email_address}));
		console.log(state);
	}
    function setContent(content: string) {
        setState(prevState => ({...prevState, content}));
		console.log(state);
    }
    function setSubj(subject_line: string) {
        setState(prevState => ({...prevState, subject_line}));
        console.log(state);
    }
    const handleChange = (event: any) => {
        console.log("ccccccccccccccc");
        console.log(event.target.value);
		setContent(event.target.value);
    }
    const handleSubjChange = (event: any) => {
       console.log(event);
        setSubj(event.target.value);
    }
    const handleEmailChange = (event: any) => {
        setEmail(event.target.value);
	}

	return (
    <div style={{ marginLeft: "30%"}}>
      	<h2>PhisherCop</h2>
		
		<div className="multiline" style={{marginRight: "40%"}}>
        <form onSubmit={handleSubmit}>
		<label className='label'>Content</label>
		<TextBoxComponent multiline={true} onChange={handleChange} input={onInput} created={onCreate} placeholder='Enter the text' floatLabelType='Never' ref = {scope => {state.textareaObj = scope }}/>
		     <label className='label'>Subject-line (optional)</label>
            <TextBoxComponent multiline={false} onChange={handleSubjChange}  placeholder='Enter the subject-line' floatLabelType='Never'/>
            <label className='label'>Email Address (optional)</label>
            <TextBoxComponent multiline={false} onChange={handleEmailChange} placeholder='Enter the email address' floatLabelType='Never'/> 
		<input type="submit" value="Submit" style={{marginLeft: "40%"}} />
		</form>
		
		<Chart 
  			width={'900px'}
  			height={'500px'}
  			chartType="PieChart"
  			loader={<div>Loading Chart</div>}
  			data={[
   			 ['Task', 'Hours per Day'],
			['Not Phishing',100 - phishing],
    		['Phishing', phishing],
  			]}
  			options={{
    		title: 'Is it phishing?',
		    PieChart: { width: '50%', height: '70%' },
  			}}
  			rootProps={{ 'data-testid': '1' }}
		/>
	 </div>
    </div>
  );
}

export default App;
