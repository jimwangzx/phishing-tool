import {POST_PHISHING, ERRORS} from '../types'
import axios from 'axios'

export const postPhishing = (text) => async dispatch => {

	try {
		const rest = await axios.post('http://localhost:4433/phishing/text', text)
		dispatch({
			type: POST_PHISHING,
			payload: rest.data
		})
	}
	catch(e) {
		dispatch({
			type: ERRORS,
			payload: console.log(e)
	})
	}
}
		
