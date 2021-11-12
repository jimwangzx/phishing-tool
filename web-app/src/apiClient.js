import axios from 'axios';

const BASE_URI = 'http://localhost:4433';

const client = axios.create({
 baseURL: BASE_URI,
 json: true
});

class APIClient {
  constructor(accessToken) {
    this.accessToken = accessToken;
  }

  doText(text) {
    return this.perform('post', '/phishing/text', text);
  }

  async perform (method, resource, data) {
    return client({
      method,
      url: resource,
      data,
    }).then(resp => {
      return resp.data ? resp.data : [];
    })
  }
}

export default APIClient;
